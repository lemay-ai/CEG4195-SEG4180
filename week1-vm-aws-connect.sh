# container: none (PowerShell and VM shell commands, typed by hand; not run by the smoke tests)
# On Windows, in PowerShell: lock down the key file, then tunnel to Jupyter
icacls my-aws.pem /inheritance:r
icacls my-aws.pem /grant:r "$($env:USERNAME):(R)"
icacls my-aws.pem /remove "Authenticated Users" "BUILTIN\Users" "BUILTIN\Administrators"
ssh -i my-aws.pem -L 8888:localhost:8888 ubuntu@3.99.169.227
# On the VM, in bash
docker pull tensorflow/tensorflow:2.21.0-gpu-jupyter
sudo bash
docker run --gpus all -it -d --name tensorflow_jupyter -p 8888:8888 tensorflow/tensorflow:2.21.0-gpu-jupyter
ps ax|grep jupyter
docker ps
docker exec -it tensorflow_jupyter bash
jupyter notebook list
