#! /bin/bash

# CK installation script for yad2k sources and demo
#
# Developer(s):
#  * Vladislav Zaborovskiy, vladzab@yandex.ru
#
# INSTALL_DIR
# YAD2K_URL
# WEIGHTS_URL
# CFG_URL

echo ""
echo "Removing everything from $INSTALL_DIR..."
rm -rf $INSTALL_DIR/*

######################################################################################
echo ""
echo "Downloading yad2k..."
git clone $YAD2K_URL $INSTALL_DIR
if [ "${?}" != "0" ] ; then
    echo ""
    echo "Error: Downloading to $INSTALL_DIR failed! Try to run installation script again"
    exit 1
fi

######################################################################################
#Demo installation
echo ""
echo "Downloading demo weights and model config..."
if [ $INSTALL_DEMO == "yes" ] ; then
    wget $WEIGHTS_URL -P $INSTALL_DIR
    if [ "${?}" != "0" ] ; then
        echo ""
        echo "Error: Downloading demo weights from $WEIGHTS_URL failed!"
        exit 1
    fi
    wget $CFG_URL -P $INSTALL_DIR
    if [ "${?}" != "0" ] ; then
        echo ""
        echo "Error: Downloading demo configuration file from $CFG_URL failed!"
        exit 1
    fi
fi

######################################################################################
echo ""
echo "Successfully finished yad2k installation."
exit 0
