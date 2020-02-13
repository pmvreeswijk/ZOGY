#!/bin/bash

# Bash script to help install ZOGY automatically on an Ubuntu
# machine. It has been tested using a Google cloud VM instance with a
# fresh install of Ubuntu 18.04 LTS.
#
# to run: download and execute "./install_zogy_ubuntu.sh"
#
# Still to do:
#
# - add the calibration binary fits catalog used by zogy
# - try to make apt-get install PSFEx (and SExtractor) with multi-threading
#
#
# versions/settings
# ================================================================================

# python version
v_python="3.7"
# zogy; for latest version, leave these empty ("") or comment out
v_zogy="0.9.1"

# define home of zogy
zogyhome=${PWD}/ZOGY

# exit script if zogyhome already exists
if [ -d "${zogyhome}" ]
then
    echo "${zogyhome} already exists; exiting script"
    exit
fi


# check Linux version, update/upgrade it and set package manager
# ================================================================================

uname_str=$(uname -a)
if [[ ${uname_str,,} == *"ubuntu"* ]]
then
    packman="apt-get"
    # update
    sudo ${packman} -y update
fi
# upgrade
sudo ${packman} -y upgrade


# python, pip and git
# ================================================================================

sudo ${packman} -y install python${v_python} python${v_python}-dev
if [ ${v_python} \< "3" ]
then
    sudo ${packman} -y install python-pip
else
    sudo ${packman} -y install python3-pip
fi
pip="python${v_python} -m pip"

# git
sudo ${packman} -y install git


# clone ZOGY repository in current directory
# ================================================================================

if [ ! -z ${v_zogy} ]
then
    zogy_branch="--branch v${v_zogy}"
    v_zogy_git="@v${v_zogy}"
fi
git clone ${zogy_branch} https://github.com/pmvreeswijk/ZOGY


# install ZOGY repository
# ================================================================================

sudo -H ${pip} install git+git://github.com/pmvreeswijk/ZOGY${v_zogy_git}


# packages used by ZOGY
# ================================================================================

# Astrometry.net
sudo ${packman} -y install astrometry.net

# SExtractor (although it seems already installed automatically)
sudo ${packman} -y install sextractor
# the executable for this installation is 'sextractor' while ZOGY
# expects 'source-extractor'; make a symbolic link
sudo ln -s /usr/bin/sextractor /usr/bin/source-extractor

# SWarp
sudo ${packman} -y install swarp
# the executable for this installation is 'SWarp' while ZOGY expects
# 'swarp'; make a symbolic link
sudo ln -s /usr/bin/SWarp /usr/bin/swarp

# PSFEx - this basic install does not allow multi-threading
sudo ${packman} -y install psfex

# ds9
sudo ${packman} -y install saods9


# download calibration catalog
# ================================================================================

url="https://storage.googleapis.com/blackbox-auxdata"

# with Kurucz templates
sudo wget -nc $url/photometry/ML_calcat_kur_allsky_ext1deg_20181115.fits.gz -P ${ZOGYHOME}/CalFiles/
# with Pickles templates
#sudo wget -nc $url/photometry/ML_calcat_pick_allsky_ext1deg_20181201.fits.gz -P ${ZOGYHOME}/CalFiles/
sudo gunzip ${ZOGYHOME}/CalFiles/ML_calcat*.gz


# download astrometry.net index files
# ================================================================================

# make sure index files are saved in the right directory; on mlcontrol
# these are in /usr/local/astrometry/data/ (config file:
# /usr/local/astrometry/etc/astrometry.cfg) while on GCloud VM
# installation they are supposed to be in /usr/share/astrometry
# (config file: /etc/astrometry.cfg)
dir1="/usr/share/astrometry"
dir2="/usr/local/astrometry/data"
dir3="${HOME}/IndexFiles"
if [ -d "${dir1}" ]
then
    dir_save=${dir1}
elif [ -d "${dir2}" ]
then
    dir_save=${dir2}
else
    dir_save=${dir3}
    mkdir ${dir3}
fi
echo "downloading Astrometry.net index files to directory ${dir_save}"
echo 
sudo wget -nc $url/astrometry/index-500{4..6}-0{0..9}.fits -P ${dir_save}
sudo wget -nc $url/astrometry/index-500{4..6}-1{0..1}.fits -P ${dir_save}


# set environent variables:
# ================================================================================

echo
echo "======================================================================"
echo
echo "copy and paste the commands below to your shell startup script"
echo "(~/.bashrc, ~/.cshrc or ~/.zshrc) for these system variables"
echo "and python alias to be set when starting a new terminal, e.g.:"
echo
echo "# ZOGY system variables"

if [[ ${SHELL} == *"bash"* ]] || [[ ${SHELL} == *"zsh"* ]]
then
    echo "export ZOGYHOME=${zogyhome}"
    echo "if [ -z \"\${PYTHONPATH}\" ]"
    echo "then"
    echo "    export PYTHONPATH=${zogyhome}:${zogyhome}/Settings"
    echo "else"
    echo "    export PYTHONPATH=\${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings"
    echo "fi"
    echo
    echo "# python alias"
    echo "alias python=python${v_python}"
fi

if [[ ${SHELL} == *"csh"* ]]
then
    echo "setenv ZOGYHOME ${zogyhome}"
    echo "if ( \$?PYTHONPATH ) then"
    echo "    setenv PYTHONPATH \${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings"
    echo "else"
    echo "    setenv PYTHONPATH ${zogyhome}:${zogyhome}/Settings"
    echo "endif"
    echo 
    echo "# python alias"
    echo "alias python python${v_python}"
fi

echo
echo "======================================================================"
echo
