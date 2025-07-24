This transcribes MP4s by converting them to WAV with FFMPEG.
It is designed to work with a webcam that's configured to save 1 minute files to a shared folder next to this running process.
It monitors all folders under a root folder looking for MP4s without correcponding TXT files.
It then uses Whisper Ivrit models on a MP4 to transcribe it.

# Installation:
## Upgrade OS
```
sudo apt update
sudo apt upgrade
sudo restart
sudo reboot
```

## Samba installation in order to configure dump from webcam
```
sudo apt install samba samba-common-bin -y
mkdir -p ~/share
chmod 777 ~/share
cd ~/share/
```
## Configure SAMBA share:
Open /etc/samba/smb.conf and under ```[global]``` put:
```
#### added support for xiaomi
server min protocol = NT1
ntlm auth = yes
map to guest = Bad user
```

Then at the end, configure the shared folders - once for xiaomi as public and once for windows as secured
***IMPORTANT: replace proper path + user names***

```
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
```

Restart samba `sudo systemctl restart smbd`

Activate samba user
```
sudo smbpasswd -a yosi
sudo smbpasswd -e yosi
```

Restart samba `sudo systemctl restart smbd`

# Install python
```
sudo apt install python3
```

# Add python alias to bashrc
```
sudo pico ~/.bashrc
```
Add following at the file end:
`alias python=python3`

# Restart shell
```
logout
```

# Check python
```
python --version
```

# Create folder
```
mkdir whisper-runner
cd whisper-runner/
```

# Create venv
```
sudo apt install python3.13-venv
python -m venv venv
```

# Activate venv
```
source venv/bin/activate
```

# Install packages
```
pip install --upgrade pip
pip install ctranslate2 transformers ffmpeg-python soundfile protobuf faster-whisper ffmpeg
```

# Install files

1. Install project files in to whisper-runner folder...
2. Install model in to subfolder - 
Download model from 
https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2/tree/main and save it in folder `ivrit-ai-whisper-large-v3-turbo-ct2`


# Run without interruption
```
source venv/bin/activate
nohup python /home/yosi/share/transcribe.py /home/yosi/share/
```

The new videos should be saved by webcam to subfolders of /share/

# View live log
tail -f transcribe.log