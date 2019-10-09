# This script is to download all 8 dataset

echo "OpenSubtitles v2018"

mkdir -p ./data/opensubtitle_v2018
wget http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-th.txt.zip  -O ./data/opensubtitle_v2018/en-th.txt.zip
unzip ./data/opensubtitle_v2018/en-th.txt.zip -d ./data/opensubtitle_v2018

echo "JW300 v1 (en)"
mkdir -p ./data/jw300_v1
wget https://object.pouta.csc.fi/OPUS-JW300/v1/raw/en.zip -O ./data/jw300_v1/en.zip
unzip ./data/jw300_v1/en.zip -d ./data/jw300_v1

echo "JW300 v1 (th)"
wget https://object.pouta.csc.fi/OPUS-JW300/v1/raw/th.zip -O ./data/jw300_v1/th.zip
unzip ./data/jw300_v1/th.zip -d ./data/jw300_v1

echo "GNOME v1"
mkdir -p ./data/gnome_v1
wget https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en-th.txt.zip -O ./data/gnome_v1/en-th.txt.zip
unzip ./data/gnome_v1/en-th.txt.zip -d ./data/gnome_v1

echo "QED v2.0a"
mkdir -p ./data/qud_v02a
wget https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-th.txt.zip -O ./data/qud_v02a/en-th.txt.zip
unzip ./data/qud_v02a/en-th.txt.zip  -d ./data/qud_v02a

echo "bible-uedin v1"
mkdir -p ./data/bible_uedin_v1
wget https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/en-th.txt.zip -O ./data/bible_uedin_v1/en-th.txt.zip
unzip ./data/bible_uedin_v1/en-th.txt.zip  -d ./data/bible_uedin_v1

echo "Tanzil v1"
mkdir -p ./data/tanzil_v1
wget https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/en-th.txt.zip -O ./data/tanzil_v1/en-th.txt.zip
unzip ./data/tanzil_v1/en-th.txt.zip -d ./data/tanzil_v1

echo "KDE4 v2"
mkdir -p ./data/kde4_v2
wget https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-th.txt.zip -O ./data/kde4_v2/en-th.txt.zip
unzip ./data/kde4_v2/en-th.txt.zip -d  ./data/kde4_v2

echo "Ubuntu v14.10"
mkdir -p ./data/ubuntu_v1410
wget https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-th.txt.zip -O ./data/ubuntu_v1410/en-th.txt.zip
unzip ./data/ubuntu_v1410/en-th.txt.zip -d  ./data/ubuntu_v1410

echo "Tatoeba v20190709"
mkdir ./data/tatpeba_v20190709
wget https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/moses/en-th.txt.zip -O ./data/tatpeba_v20190709/en-th.txt.zip
unzip ./data/tatpeba_v20190709/en-th.txt.zip -d  ./data/tatpeba_v20190709