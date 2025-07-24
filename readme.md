This transcribes MP4s by converting them to WAV with FFMPEG.
It is designed to work with a webcam that's configured to save 1 minute files to a shared folder next to this running process.
It monitors all folders under a root folder looking for MP4s without correcponding TXT files.
It then uses Whisper Ivrit models on a MP4 to transcribe it.

Installation:
# upgrade OS
sudo apt update
sudo apt upgrade
sudo restart
sudo reboot

# samba installation in order to configure dump from webcam
sudo apt install samba samba-common-bin -y
mkdir -p ~/share
chmod 777 ~/share
cd ~/share/
# configure SAMBA share:
# open /etc/samba/smb.conf
# under global put:
#### added support for xiaomi
server min protocol = NT1
ntlm auth = yes
map to guest = Bad user

# then at the end configure the shared folders - once for xiaomi as public and once for windows as secured
# replace proper path + user names
[nas]
path = /home/yosi/share
writeable = yes
create mask = 0777
directory mask = 0777
valid users = yosi
guest ok = no

[share]
path = /home/yosi/share
browseable = yes
guest ok = yes
writable = yes
create mask = 0777
directory mask = 0777
force user = yosi

### end samba configuration

sudo systemctl restart smbd
# activate samba user
sudo smbpasswd -a yosi
sudo smbpasswd -e yosi
sudo systemctl restart smbd

# install python
sudo apt install python3
python3
python

# add python alias to bashrc
sudo pico ~/.bashrc
# add following at the file end:
alias python=python3

# restart shell
logout

# check python
python --version

# create folder
mkdir whisper-runner
cd whisper-runner/

# create venv
sudo apt install python3.13-venv
python -m venv venv
# activate venv
source venv/bin/activate

pip install --upgrade pip
pip install ctranslate2 transformers ffmpeg-python soundfile protobuf faster-whisper ffmpeg

# install model in to subfolder  ivrit-ai-whisper-large-v3-turbo-ct2

# install files

# run without interruption
source venv/bin/activate
nohup python /home/yosi/share/transcribe.py /home/yosi/share/

# view log
tail -f transcribe.log