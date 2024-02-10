[[_TOC_]]

# Getting Started

More information on using the cluster can be found at [the SIC wiki](https://wiki.cs.uni-saarland.de/en/HPC/faq) and [in this tutorial](https://kingsx.cs.uni-saarland.de/index.php/s/ssmj33dxmgGsAYd).

## Connecting to the cluster

To connect to the cluster you need to be in the university's network. If you are at home you can use the [VPN](https://www.hiz-saarland.de/dienste/vpn/).

### Standard SSH Connection
To initiate a connection with the cluster, execute the following SSH command, substituting `<username>` with your specific username:

```bash
ssh <username>@conduit.cs.uni-saarland.de
```

Upon execution, enter your assigned password when prompted to complete the login process.

### Utilizing SSH Keys for easier access

#### Step 1: Generating an SSH Key
To avoid entering your password on each login, consider setting up SSH keys. Start by generating a new SSH key. Note these commands should be run on your local machine not the cluster.
```bash
ssh-keygen -t rsa -b 4096 -f .ssh/sic_cluster
```

#### Step 2: Transferring the Public Key to the Cluster
Next, transfer the newly created public key to the cluster to enable key-based authentication:

```bash
ssh-copy-id -i ~/.ssh/sic_cluster.pub <username>@conduit.cs.uni-saarland.de
```

Post completion, you should be able to log in using `ssh <username>@conduit.cs.uni-saarland.de` without entering the password each time.

#### Simplifying the SSH Command
To further simplify the SSH connection process, you can add an entry to your `~/.ssh/config` file:

```bash
Host sic_cluster
    HostName conduit.cs.uni-saarland.de
    User <username>
```
Replace `<username>` with your specific username (`neuronet_teamxyz`). With this configuration, you can connect to the cluster by simply executing `ssh sic_cluster`.

## Setting up on the cluster
Once you are able to connect to the cluster, you can get started by cloning or even better forking this repository.
```
git clone https://gitlab.cs.uni-saarland.de/mara00002/torch-condor-template
# or git clone your fork
```
To makes changes you can either develop on the cluster directly using e.g. VSCode with the  [Remote-SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension or a terminal editor, or setup your own repository and push changes from your machine. The second strategy is a little more involved, but allows you to develop using your favorite programs on your local machine.

### Developing using git


1. **Set up your own remote**. You can use your [SIC account](https://sam.sic.saarland/) on the SIC's GitLab instance to create a [new repository](https://gitlab.cs.uni-saarland.de/projects/new#blank_project). Any other hosted git instance of course works as well. 
2. **Change the remote to your repo**. To change the remote run `git remote set-url origin https://url-to-your-repo.git`. 
3. **Push and Pull**.
```bash
# On the remote/cluster
git push
``` 
```bash
# On your machine 
git clone https://url-to-your-repo.git
```
Now you can make changes on your machine and simply push to the repo and pull the changes on the cluster.

## Installation of Miniconda

### Step 1: Install Miniconda
To initiate the setup, begin by installing Miniconda. Execute the following command:

```bash
condor_submit setup.sub
```

This command installs Miniconda in the directory `~/miniconda3` and includes all the packages listed in `environment.yml` into an environment with the name `nnti`. You can change the name in the `environment.yml` file.

Should you require additional packages, you can easily incorporate them by adding them to the `environment.yml` file and re-executing the setup job.

### Step 2: Configuring the System Path (Optional)
To integrate `conda` into your system path, append the following line to your `~/.bashrc` file:

```bash
export PATH=$HOME/miniconda3/bin:$PATH
```

Afterwards, activate the changes by sourcing the `~/.bashrc` file:

```bash
source ~/.bashrc
```

### Step 3: Submitting Scripts to the Cluster
With Miniconda configured, you can now submit scripts to the cluster. For instance, to submit the `torch_matmul_docker.py` script, use the following command:

```bash
condor_submit run.sub
```

This process triggers `conda_run.sh`, which in turn selects the Conda environment as specified in the `environment.yml` file, and subsequently executes the script defined in `run.sh`. To modify the Python script being executed, simply edit the last line in the `run.sh` file.

## Monitoring Job Execution and Logs

### Accessing Job Logs
Upon the successful execution of a job, its log files are stored within the `logs/` directory. To review the output of a specific job, utilize the following command, replacing `<job_id>` with the actual ID of the job:

```bash
less logs/run.<job_id>.0.out
```

### Real-time Monitoring of Ongoing Processes
For real-time monitoring of a process that is currently in execution, the following command can be used, again substituting `<job_id>` with the correct job ID:

```bash
tail -f logs/run.<job_id>.0.out
```
## How to View Files on the Cluster

Since the cluster does not have a graphical user interface (GUI), you need alternative methods to view files, such as plots. You can either transfer these files to your local computer or use an application that provides a GUI through SSH.

### For Linux Users
If you're using Linux, your file explorer likely supports SFTP. You can access and view all files on the cluster by connecting to `sftp://conduit.cs.uni-saarland.de`. This allows you to browse the cluster's files directly from your file explorer.

### For Windows Users
On Windows you can e.g. use a file transfer application with SFTP support, such as  [FileZilla](https://filezilla-project.org/) or [WinSCP](https://winscp.net/eng/download.php). These programs offer a graphical interface for transferring files from the cluster to your computer. 

### Using Visual Studio Code
Regardless of your operating system, you can use Visual Studio Code with the [Remote-SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension. This setup allows you to develop directly on the cluster. It also enables you to view files, including images, as if you were working locally.

### Copying Files Using `scp`
Another method is to copy the files to your local machine using the `scp` command. This is useful for viewing files on your own computer.

To copy a file from the cluster to your machine, use the following command. Replace `<username>` with your actual username, `/path/to/your/file/on/remote` with the file's path on the cluster, and `/copy/to/here/` with the destination path on your machine.

```bash
scp <username>@conduit.cs.uni-saarland.de:/path/to/your/file/on/remote /copy/to/here/
```

If you have previously set up an entry in your `~/.ssh/config` file, the command simplifies to:

```bash
scp sic_cluster:/path/to/your/file/on/remote /copy/to/here/
```

This command uses the alias `sic_cluster` that you would have defined in your SSH configuration.


### Git
If you have setup your own repository, you can also add the files to your repository.

