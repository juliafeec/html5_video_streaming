from user_definition import *
from os.path import expanduser
import paramiko
import warnings
warnings.filterwarnings("ignore")


def ssh_client():
    """Return ssh client object"""
    return paramiko.SSHClient()


def ssh_connection(ssh, ec2_address, user, key_file):
    """Establish ssh connection"""
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ec2_address, username=user,
                    key_filename=expanduser("~") + key_file)
    except TimeoutError:
        return None

    return ssh


def id_rsa_gen(ssh, id_rsa):
    """generate id_rsa key, move it to the right location
    and make it private"""
    ftp = ssh.open_sftp()
    file = ftp.file('id_rsa', "a", -1)
    file.write(id_rsa)
    file.flush()
    ftp.close()
    ssh.exec_command("mv id_rsa ~/.ssh/id_rsa")
    ssh.exec_command("cd .ssh/ \n chmod 400 id_rsa")


def id_rsa_pub_gen(ssh, id_rsa_pub):
    """generate id_rsa public key, move it
    to the right location and make it private"""
    ftp = ssh.open_sftp()
    file = ftp.file('id_rsa.pub', "a", -1)
    file.write(id_rsa_pub)
    file.flush()
    ftp.close()
    ssh.exec_command("mv id_rsa.pub ~/.ssh/id_rsa.pub")
    ssh.exec_command("cd .ssh/ \n chmod 400 id_rsa.pub")


def aws_gen(ssh, aws_info):
    """turn off ansible such that git clone won't throw error"""
    ftp = ssh.open_sftp()
    file = ftp.file('credentials', "a", -1)
    file.write(aws_info)
    file.flush()
    ftp.close()
    ssh.exec_command("mkdir -p ~/.aws/")
    ssh.exec_command("mv credentials ~/.aws/credentials")


def git_clone_pull(ssh, ssh_rsa):
    """git clone the repo and if there is a repo pull it"""
    ssh.exec_command("""echo "%s" >> ~/.ssh/known_hosts""" % ssh_rsa)
    stdin, stdout, stderr = ssh.exec_command(
        "git clone git@github.com:MSDS698/Pelicam.git")
    if (b"already exists" in stderr.read()):
        stdin, stdout, stderr = ssh.exec_command(
            "cd Pelicam; git pull git@github.com:MSDS698/Pelicam.git")
        stderr.read()


def miniconda(ssh):
    """setup minicode"""
    ssh.exec_command("wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh")
    ssh.exec_command("chmod +x Miniconda3-latest-Linux-x86_64.sh")
    stdin, stdout, stderr = ssh.exec_command("./Miniconda3-latest-Linux-x86_64.sh -b")
    if (b"already exists" in stderr.read()):
        stdin, stdout, stderr = ssh.exec_command("./Miniconda3-latest-Linux-x86_64.sh -b -u")
        # have to call .read() to update, need more insight
        stderr.read()


def create_or_update_environment(ssh):
    """Setup Anaconda environment"""
    sstdin, stdout, stderr = ssh.exec_command("export PATH=\"/home/ubuntu/miniconda3/bin:$PATH\" && conda env create -f ~/Pelicam/environment.yml", get_pty=True)

    if (b'already exists' in stderr.read()):
        sstdin, stdout, stderr = ssh.exec_command("export PATH=\"/home/ubuntu/miniconda3/bin:$PATH\" && conda env update -f ~/Pelicam/environment.yml", get_pty=True)


def install(ssh):
    """install needed packages"""
    ssh.exec_command("sudo apt-get install -y python-opencv", get_pty=True)
    ssh.exec_command("sudo apt-get -y install run-one", get_pty=True)
    ssh.exec_command("sudo apt -y install awscli", get_pty=True)


def s3_model(ssh):
    """download models"""
    ssh.exec_command(
        "aws s3 cp s3://msds603camera/extracted_dict.pkl ~/Pelicam/code/", get_pty=True)
    stdin, stdout, stderr = ssh.exec_command(
        "git clone https://github.com/davidsandberg/facenet", get_pty=True)
    if (b"already exists" in stderr.read()):
        ssh.exec_command("git pull https://github.com/davidsandberg/facenet", get_pty=True)
    ssh.exec_command("export PATH=\"/home/ubuntu/miniconda3/bin:$PATH\" && source activate MSDS603 && pip install https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py36/tensorflow-1.9.0-cp36-cp36m-linux_x86_64.whl", get_pty=True)


def setup_server(ssh):
    """download servers"""
    ssh.exec_command("sudo apt-get -y install redis-server", get_pty=True)
    ssh.exec_command("sudo systemctl enable redis-server.service", get_pty=True)
    ssh.exec_command("sudo systemctl restart redis-server.service", get_pty=True)
    ssh.exec_command("sudo apt -y remove unattended-upgrades", get_pty=True)


def start_server(ssh):
    """running servers"""
    ssh.exec_command("""export PATH=\"/home/ubuntu/miniconda3/bin:$PATH\" && source activate MSDS603 && \
        tmux new -d -s server -n ffserver /usr/local/bin/ffserver -f /home/ubuntu/Pelicam/code/ffserver.conf; new-window -n "ffmpeg" -d "run-one-constantly \
        /home/ubuntu/Pelicam/code/start_ffmpeg.sh &> \
        /home/ubuntu/log_ffmpeg.txt" \
        ; new-window -n "flask" -d \
        "/home/ubuntu/miniconda3/envs/MSDS603/bin/python \
        /home/ubuntu/Pelicam/code/app.py" \
        ; new-window -n "tail" -d "tail -f /home/ubuntu/log_ffmpeg.txt" \
        ; new-window -n "bash" -d "bash" \
        ; next-window \
        ; next-window \
        ; next-window \
        ;""", get_pty=True)


def main():
    """main function"""
    # initialize and login ssh
    print("setting up initial contact")
    ssh = ssh_client()
    ssh = ssh_connection(ssh, ec2_address, user, key_file)
    if ssh is None:
        # if it can't connect
        print("Can't connect to the server")
    else:
        # upadte and install basic packges
        print("acquire basic packges")
        stdin, stdout, stderr = ssh.exec_command("sudo apt update")
        stdin, stdout, stderr = ssh.exec_command(
            "sudo apt install -y htop tmux git")

        # update security files
        print("updating security files")
        id_rsa_gen(ssh, id_rsa)
        id_rsa_pub_gen(ssh, id_rsa_pub)
        aws_gen(ssh, aws_info)

        # clone or pull docs
        print("clone or pulling classroom repo")
        git_clone_pull(ssh, ssh_rsa)

        # need to initiate and then update, need more insight
        print("installing minicoda")
        miniconda(ssh)
        miniconda(ssh)

        # create env
        print("create and update environment")
        create_or_update_environment(ssh)

        # necessary further install
        print("install necessary packages")
        install(ssh)

        # clone models and embeddings
        print("pulling models, weights and embeddings from s3")
        s3_model(ssh)

        # setup servers
        print("setup severs")
        setup_server(ssh)

        # run servers
        start_server(ssh)
        print("server is running")

        print("Using port: 5001")

        ssh.close()
        print("Logged out.")


if __name__ == '__main__':
    main()
