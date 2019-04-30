# install miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
export PATH=~/miniconda3/bin:$PATH
source ~/.bashrc
conda env update -f ~/Pelicam/environment.yml
conda activate MSDS603
export PATH=/home/ubuntu/miniconda3/envs/MSDS603/bin:$PATH
source ~/.bashrc
sudo apt-get install -y python-opencv
sudo apt-get -y install run-one
sudo apt -y install awscli

# get s3 data
aws s3 cp s3://msds603camera/extracted_dict.pickle ~/Pelicam/code/
aws s3 cp s3://msds603camera/ffmpeg_3.4.6-1_amd64.deb ~
aws s3 cp s3://msds603camera/ffserver_20190407-1_amd64.deb ~
aws s3 cp s3://msds603camera/20180402-114759-chris ~/Pelicam/code/20180402-114759 --recursive

# install ffmpeg, tensorflow and redis
git clone https://github.com/davidsandberg/facenet.git
sudo apt-get install ./ffmpeg_3.4.6-1_amd64.deb
sudo apt-get install ./ffserver_20190407-1_amd64.deb
pip install https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl
sudo apt-get -y install redis-server
sudo systemctl enable redis-server.service
sudo apt -y remove unattended-upgrades

# install ffmpeg dependencies
sudo apt-get -y install libsdl2-dev
sudo apt-get -y install libxv-dev
sudo apt-get -y install libass-dev
sudo apt-get -y install libvdpau-dev
sudo apt-get -y install libva-x11-1
sudo apt-get -y install libva-drm1
sudo apt-get -y install libfdk-aac0

# start servers
tmux new -s "server" -d
tmux new-window -n "ffserver" -d "/usr/local/bin/ffserver -f /home/ubuntu/Pelicam/code/remote_ffserver.conf"
tmux new-window -n "ffmpeg" -d "run-one-constantly /home/ubuntu/Pelicam/code/start_ffmpeg_remote.sh &> /home/ubuntu/log_ffmpeg.txt"
# tmux new-window -n "flask" -d "/home/ubuntu/miniconda3/envs/MSDS603/bin/python /home/ubuntu/Pelicam/code/app.py"
cd ~/Pelicam/code
tmux new-window -n "flask" -d "flask run --host=0.0.0.0 --port=5001"
tmux new-window -n "tail" -d "tail -f /home/ubuntu/log_ffmpeg.txt"
# tmux new-window -n "face detection" -d "/home/ubuntu/miniconda3/envs/MSDS603/bin/python /home/ubuntu/Pelicam/code/face_detection_process.py"