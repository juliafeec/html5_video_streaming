from os.path import expanduser
import paramiko
import warnings
import sys
sys.path.append("code")
from user_definition import *
warnings.filterwarnings("ignore")

def ssh_client():
    """Return ssh client object"""
    return paramiko.SSHClient()


def ssh_connection(ssh, ec2_address, user, key_file):
    """Establish ssh connection"""
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ec2_address,
                    username=user,
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
    ssh.exec_command("chmod 400 ~/.ssh/id_rsa")


def id_rsa_pub_gen(ssh, id_rsa_pub):
    """generate id_rsa public key, move it
    to the right location and make it private"""
    ftp = ssh.open_sftp()
    file = ftp.file('id_rsa.pub', "a", -1)
    file.write(id_rsa_pub)
    file.flush()
    ftp.close()
    ssh.exec_command("mv id_rsa.pub ~/.ssh/id_rsa.pub")
    ssh.exec_command("chmod 400 ~/.ssh/id_rsa.pub")


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
        tmux_install = "sudo apt install -y htop tmux git"
        stdin, stdout, stderr = ssh.exec_command("sudo apt update")
        stdin, stdout, stderr = ssh.exec_command(tmux_install)

        # update security files
        print("updating security files")
        id_rsa_gen(ssh, id_rsa)
        id_rsa_pub_gen(ssh, id_rsa_pub)
        aws_gen(ssh, aws_info)

        # clone or pull docs
        print("clone or pulling classroom repo")
        git_clone_pull(ssh, ssh_rsa)

        print("install remaining packages and start up servers")
        ssh.exec_command("chmod +x ~/Pelicam/start_up.sh")
        ssh.exec_command("~/Pelicam/start_up.sh > start_up_log")

        print("Using port: 5001")

        ssh.close()
        print("Logged out.")


if __name__ == '__main__':
    main()
