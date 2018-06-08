#pragma once

#include <stdio.h>
#include <setjmp.h>
#include <jpeglib.h>
#include <math.h>

struct ImageData {
  std::vector<uint8_t> data;
  int height;
  int width;
  int channels;
};

// Error handling for JPEG decoding.
void catch_jpeg_error(j_common_ptr cinfo) {
  (*cinfo->err->output_message)(cinfo);
  jmp_buf *jpeg_jmpbuf = reinterpret_cast<jmp_buf *>(cinfo->client_data);
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// Decompresses a JPEG file from disk.
ImageData load_jpeg_file(std::string file_name) {
  FILE *infile = fopen(file_name.c_str(), "rb");
  if (!infile)
    throw "Can't open " + file_name;

  jpeg_error_mgr jerr;
  jpeg_decompress_struct cinfo;
  jmp_buf jpeg_jmpbuf;  // recovery point in case of error
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = catch_jpeg_error;
  if (setjmp(jpeg_jmpbuf)) {
    fclose(infile);
    throw std::string("JPEG decoding failed");
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  ImageData res;
  res.width = cinfo.output_width;
  res.height = cinfo.output_height;
  res.channels = cinfo.output_components;
  res.data.resize(res.height * res.width * res.channels);

  int row_stride = cinfo.output_width * cinfo.output_components;
  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
  while (cinfo.output_scanline < cinfo.output_height) {
    uint8_t* row_address = &(res.data[cinfo.output_scanline * row_stride]);
    jpeg_read_scanlines(&cinfo, buffer, 1);
    memcpy(row_address, buffer[0], row_stride);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);  

  return res;
}

// Convert the eight-bit data in the image into float, resize
// it using bilinear filtering, and scale it numerically to the float range that
// the model expects (given by input_mean and input_std).
void resize_image(float* out, const ImageData& in,
                  int wanted_height, int wanted_width,
                  float input_mean = 128, float input_std = 128) {
  const int wanted_channels = in.channels;
  const size_t image_rowlen = in.width * in.channels;
  const float width_scale = static_cast<float>(in.width) / wanted_width;
  const float height_scale = static_cast<float>(in.height) / wanted_height;
  for (int y = 0; y < wanted_height; ++y) {
    const float in_y = y * height_scale;
    const int top_y_index = static_cast<int>(floorf(in_y));
    const int bottom_y_index = std::min(static_cast<int>(ceilf(in_y)), (in.height - 1));
    const float y_lerp = in_y - top_y_index; 
    const uint8_t* in_top_row = in.data.data() + (top_y_index * image_rowlen);
    const uint8_t* in_bottom_row = in.data.data() + (bottom_y_index * image_rowlen);
    float *out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const float in_x = x * width_scale;
      const int left_x_index = static_cast<int>(floorf(in_x));
      const int right_x_index = std::min(static_cast<int>(ceilf(in_x)), (in.width - 1));
      const uint8_t* in_top_left_pixel = in_top_row + (left_x_index * wanted_channels);
      const uint8_t* in_top_right_pixel = in_top_row + (right_x_index * wanted_channels);
      const uint8_t* in_bottom_left_pixel = in_bottom_row + (left_x_index * wanted_channels);
      const uint8_t* in_bottom_right_pixel = in_bottom_row + (right_x_index * wanted_channels);
      const float x_lerp = in_x - left_x_index;
      float *out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        const float top_left((in_top_left_pixel[c] - input_mean) / input_std);
        const float top_right((in_top_right_pixel[c] - input_mean) / input_std);
        const float bottom_left((in_bottom_left_pixel[c] - input_mean) / input_std);
        const float bottom_right((in_bottom_right_pixel[c] - input_mean) / input_std);
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        out_pixel[c] = top + (bottom - top) * y_lerp;
      }
    }
  }
}
