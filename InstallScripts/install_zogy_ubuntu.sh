#!/bin/bash

# Bash script to help install ZOGY automatically on an Ubuntu
# machine. It has been tested using a Google cloud VM instance with a
# fresh install of Ubuntu 18.04 LTS.
#
# to run: download and execute "./install_zogy_ubuntu.sh"
#
# Still to do:
#
# - try to make apt-get install PSFEx (and SExtractor) with multi-threading
#
#
# versions/settings
# ================================================================================

# python version
v_python="3"
# zogy and meercrab; for latest version, leave these empty ("") or comment out
v_zogy=""
v_meercrab=""

# define home of zogy and meercrab
zogyhome=${PWD}/ZOGY
meercrabhome=${PWD}/meercrab

# exit script if zogyhome already exists
if [ -d "${zogyhome}" ] || [ -d "${meercrabhome}" ]
then
    echo "${zogyhome} and/or ${meercrabhome} already exist(s); exiting script"
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

echo "installing python, pip and git"
sudo ${packman} -y install python${v_python}
sudo ${packman} -y install python${v_python}-dev

if [ ${v_python} \< "3" ]
then
    sudo ${packman} -y install python-pip
else
    sudo ${packman} -y install python3-pip
fi
pip="python${v_python} -m pip"

# git
sudo ${packman} -y install git git-lfs


# clone ZOGY repository in current directory
# ================================================================================

echo "cloning ZOGY repository"
if [ ! -z ${v_zogy} ]
then
    zogy_branch="--branch v${v_zogy}"
    v_zogy_git="@v${v_zogy}"
fi
git clone ${zogy_branch} https://github.com/pmvreeswijk/ZOGY


if [ ! -z ${v_meercrab} ]
then
    meercrab_branch="--branch v${v_meercrab}"
    v_meercrab_git="@v${v_meercrab}"
fi
echo "cloning meercrab repository"
git clone ${meercrab_branch} https://github.com/Zafiirah13/meercrab
cd ${meercrabhome}
git lfs install
git lfs pull
cd ${meercrabhome}/..


# install ZOGY and MeerCRAB repositories
# ================================================================================

echo "installing ZOGY packages"
sudo -H ${pip} install git+git://github.com/pmvreeswijk/ZOGY${v_zogy_git}

echo "installing MeerCRAB packages"
# for MeerCRAB, not possible to use setup.py on git with latest python:
#sudo -H ${pip} install git+git://github.com/Zafiirah13/meercrab${v_meercrab_git}
# so install required packages manually:
sudo -H ${pip} install pandas tensorflow imbalanced-learn matplotlib scipy keras Pillow scikit_learn numpy astropy h5py==2.10.0 testresources


# packages used by ZOGY
# ================================================================================

# Astrometry.net
echo "installing astrometry.net"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install astrometry.net

# SExtractor (although it seems already installed automatically)
echo "installing sextractor"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install sextractor
# the executable for this installation is 'sextractor' while ZOGY
# versions starting from 0.9.2 expect 'source-extractor'; make a
# symbolic link; N.B.: since 2020-04-25 not needed anymore
#sudo ln -s /usr/bin/sextractor /usr/bin/source-extractor

# SWarp
echo "installing SWarp"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install swarp
# the executable for this installation is 'SWarp' while ZOGY expects
# 'swarp'; make a symbolic link
sudo ln -s /usr/bin/SWarp /usr/bin/swarp

# PSFEx - this basic install does not allow multi-threading
echo "installing PSFEx"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install psfex

# ds9; add environment DEBIAN_FRONTEND to avoid interaction with TZONE
echo "installing saods9"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install saods9


# download calibration catalog
# ================================================================================

url="https://storage.googleapis.com/blackbox-auxdata"

# with Kurucz templates
sudo wget -nc $url/photometry/ML_calcat_kur_allsky_ext1deg_20181115.fits.gz -P ${ZOGYHOME}/CalFiles/
# with Pickles templates
#sudo wget -nc $url/photometry/ML_calcat_pick_allsky_ext1deg_20181201.fits.gz -P ${ZOGYHOME}/CalFiles/
echo "gunzipping calibration catalog(s) ..."
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

# let /usr/bin/python refer to version installed above
sudo ln -sf /usr/bin/python${v_python} /usr/bin/python

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
    echo "export MEERCRABHOME=${meercrabhome}"   
    echo "if [ -z \"\${PYTHONPATH}\" ]"
    echo "then"
    echo "    export PYTHONPATH=${zogyhome}:${zogyhome}/Settings:${meercrabhome}"
    echo "else"
    echo "    export PYTHONPATH=\${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings:${meercrabhome}"
    echo "fi"
fi

if [[ ${SHELL} == *"csh"* ]]
then
    echo "setenv ZOGYHOME ${zogyhome}"
    echo "setenv MEERCRABHOME ${meercrabhome}"   
    echo "if ( \$?PYTHONPATH ) then"
    echo "    setenv PYTHONPATH \${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings:${meercrabhome}"
    echo "else"
    echo "    setenv PYTHONPATH ${zogyhome}:${zogyhome}/Settings:${meercrabhome}"
    echo "endif"
fi

echo
echo "======================================================================"
echo
