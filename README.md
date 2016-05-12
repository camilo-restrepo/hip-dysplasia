# imagenes

## Requirements

  * xcode 5.1.1
  * python 3.4.0
  * Qt libraries 5.2.1
  * SIP 4.15.5
  * PyQt 5.2.1

## Download

  * [xcode 5.1.1](https://developer.apple.com/xcode/downloads/)
  * [python 3.4.0](https://www.python.org/download/)
  * [Qt libraries 5.2.1](http://qt-project.org/downloads)
  * [SIP 4.15.5](http://www.riverbankcomputing.co.uk/software/sip/download)
  * [PyQt 5.2.1](http://www.riverbankcomputing.co.uk/software/pyqt/download5)

## installation

  * install xcode
  * install the Command Line Tools (open Xcode > Preferences > Downloads)
  * install Qt libraries (qt-opensource-mac-x64-clang-5.2.1.dmg)
  * install python 3.4
  * create a virtual env (i.e. ~/.env/ariane_mail)
  * unzip and compile SIP and PyQt

```
    cd /var/tmp
    cp /Users/gvincent/Downloads/PyQt-gpl-5.2.1.tar.gz .
    cp /Users/gvincent/Downloads/sip-4.15.5.tar.gz .
    tar xvf PyQt-gpl-5.2.1.tar.gz
    tar xvf sip-4.15.5.tar.gz
    cd sip-4.15.5/
    python3 configure.py -d ~/.env/ariane_mail/lib/python3.4/site-packages --arch x86_64
    make
    sudo make install
    sudo make clean
    cd ../PyQt-gpl-5.2.1/
    python3 configure.py --destdir ~/.env/ariane_mail/lib/python3.4/site-packages --qmake ~/Qt5.2.1/5.2.1/clang_64/bin/qmake
    make
    sudo make install
    sudo make clean
    ~/.env/ariane_mail/bin/python -c "import PyQt5"
```
