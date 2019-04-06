# README

## AWS SSH Tunnel Setup

#### First we need to enable an option in the SSH server to allow us to expose a port to external IPs:

These steps were done in Ubuntu Server 16.04

SSH into AWS and do:
`sudo vi /etc/ssh/sshd_config` (or `sudo nano /etc/ssh/sshd_config` if you prefer nano)

Add line below to the file and then save it:

`GatewayPorts clientspecified`

Then run `sudo service sshd restart`

Additionally, open inbound ports 8080 and 5001 on AWS.

#### Now we do the tunneling:

From the camera server, SSH into AWS EC2 again with

`ssh -i YOUR_KEY.pem -R 0.0.0.0:8080:0.0.0.0:5000 ubuntu@YOUR_EC2_PUBLIC_IP`

This will create a reverse tunnel so that connecting (inbound) on port 8080 on AWS will redirect to port 5000 on the camera server.
This is equivalent to port forwarding port 5000 on the camera server network. Any IP will be able to access it. 

This connection needs to be kept open the entire time, otherwise the tunneling will stop.

You can only run one of the two versions below at a time.

## Motion JPEG

1 - On the camera server, run `python app.py` inside **motion_jpeg/flask_app_camera**. This will start the server on port 5000.

In the camera server, you can go to `localhost:5000` and you should see the stream.

Make sure the SSH tunneling is running for the remote version to work.
Now you can go to `YOUR_EC2_PUBLIC_IP:8080` and you should see the same stream again, but this time it's going through AWS so there's some lag.

2 - On AWS, run `python app.py` inside **motion_jpeg/flask_app_camera_ip**. This will start the server on port 5001. 

This app receives the frames from 1, overlays it and resends it.

Now you can go to `YOUR_EC2_PUBLIC_IP:5001` and see the overlayed stream.

## Webm

First install ffmpeg. This code relies on ffserver, which is no longer available in very recent versions of ffmpeg, so make sure you get something up to 3.4.x. 
I'm using ffmpeg 3.4.4.

I also had to install opencv using `sudo apt-get install python-opencv` (this will be different for a Mac) and `pip install opencv-contrib-python` 
(just `pip install opencv` didn't work for me)

1 - On the camera server, run `ffserver -f ffserver.conf`. Leave this running.

2 - On another terminal tab, run `./start_ffmpeg.sh`. Leave this running.

Note: I've only tested this on linux and this command might need changes to run on a Mac. I suspect you need to change `-f v4l2` into `-f avfoundation -i "default"` in the `start_ffmpeg.sh` file.
If that doesn't work, try referring to the docs: https://trac.ffmpeg.org/wiki/Capture/Webcam

3 - On another terminal tab, run `python app.py` inside **webm/flask_app_camera**.

Now if you go to `localhost:5000` you should see the stream. 
Also, if you go to `YOUR_EC2_PUBLIC_IP:8080`, you should see the same stream again, this time going through AWS.

## Usage

### Test Locally On Mac
First you need to install required dependencies

`$conda install ffmpeg=3.4` yes

`$ffmpeg` in your terminal
make sure it is 3.4 or 3.4.x and using conda ffmpeg

Open a new terminal

`$cd webm/flask_app_camera` folder
`$ffserver -f ffserver.conf`

Open a new terminal
`$cd webm/flask_app_camera` folder
`$./start_ffmpeg.sh`

Open a new terminal
`$cd webm/flask_app_camera` folder

`$pip install svgwrite`,if you don’t have it

`$python app.py`

Open a new tab in browser
Go http://localhost:5000
