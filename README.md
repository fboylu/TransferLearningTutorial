Copyright (C) Microsoft Corporation.

# Image Classification Using Transfer Learning

Image classification is one of the most popular areas of deep learning and there are many real world business scenarios that benefit from its application. As a real world example, one application area is detecting quality issues on assembly lines during the manufacturing process. 

In a typical production line, components travel down the assembly line from one station to the other and at the end of each station they are analyzed by human inspectors to spot any problems. However, this is a very manual process which involves a lot of human effort which may in turn lead to delays in the production lines. By using image classification, it is possible to reduce human effort and deploy a system that automatically classifies images of components into pass and fail. In turn, this can greatly speed up the efficiency of the human operators in their validation process and improves the quality of the overall manufacturing process. 

This example highlights the transfer learning approach which uses a pretrained model to generate visual features of images which are later used to train a binary classifier. You can run this example by importing your own image data assuming you have two classes of images or you can use a generic dataset as will be descibed in the step-by-step instructions.

In this example, you will train your models through Azure Machine Learning Workbench’s Jupyter notebook service. You will be using GPU power on Linux Data Science Virtual Machines through containerized remote execution target capability provided by Azure Machine Learning Workbench. You will then operationalize your models on local docker and also on Azure Container Services Cluster and consume the web services by scoring your images in real time.

You can start by following the step-by-step instructions below.

# Prerequisites
 You will need an [Azure account](https://azure.microsoft.com/en-us/free/) (free trials are available) to perform the following tasks in the instructions below.
 - You will provision Azure Machine Learning Accounts. 
- You will create a Data Science Virtual Machine.
- You will creat an Azure Storage Account to host your training data.

# Create Azure Machine Learning Accounts
In order to use Azure Machine Learning Workbench, you will first provision Azure Machine Learning accounts using the Azure portal. 

1. Open your web browser and go to the [Azure portal](https://portal.azure.com/).
2. Follow the steps under [Create Azure Machine Learning Accounts](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation#create-azure-machine-learning-accounts) to provision the experimentation and model management accounts.
3. If you don't already have Azure Machine Learning Workbench installed, follow the steps under [Install Azure Machine Learning Workbench on Windows](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation#install-azure-machine-learning-workbench-on-windows) to complete the installation.
4. Find the shortcut to Azure Machine Learning Workbench on your desktop to start the app.
5. Sign in to Workbench by using the same account that you used earlier to provision your Azure Machine Learning accounts.
6. When the sign-in process has succeeded, Workbench attempts to find the Machine Learning Experimentation accounts that you created earlier. It searches for all Azure subscriptions to which your credential has access. When at least one Experimentation account is found, Workbench opens with that account. It then lists the workspaces and projects found in that account.

# Creating a New Project from Visual Studio Team Services GIT Repository

You will download the project from Visual Studio Team Services GIT Repository. In Workbench, follow the steps below:

1. Select **File > New Project** (or select the + sign in the **PROJECTS** pane).
2. Fill in the **Project name** and **Project directory** boxes. **Project description** is optional but helpful. Use the following URL for **Visualstudio GIT Repository URL** box: "URL for the project repository here!!"
3. Select the **Create** button to create the project. A new project is created and opened for you. At this point, you can explore the project home page, notebooks, and other files.

# Login to Azure Account and Set Subscription
1. From your Azure Machine Learning Workbench, open a Command Line Interface (CLI) by clicking **File -> Open Command Prompt**. 
2. From the CLI, log in to your Azure account by running the following command:
```bash
az login
```
3. You will be asked to visit a URL and type in a provided temporary code, the website will request your Azure account credentials. 
4. Run the following to see a list of subscriptions available in your Azure account.
```bash
az account list -o table
```
4. Find the "name" of the subscription you would like to use and insert it where indicated in the following command and run it to set the active subscription.
```bash
az account set --subscription "Your Subscription Name Here!!"
```
5. Verify that your account is set by running the following command.
```bash
az account show
```

# Configuring Execution in Docker Environment on Linux Data Science Virtual Machines with GPUs
Main notebooks of this scenario will be executed in *docker* environment on a Linux Data Science Virtual Machine (DSVM) with GPUs. While it is possible to use GPUs on any Linux virtual machine, the Ubuntu-based DSVM comes with the required CUDA drivers and libraries pre-installed, making the set-up much easier. If you don't already have one, follow the below steps to create one.

## Create an Ubuntu-based Linux DSVM with GPU
1. Open your web browser and go to the [Azure portal](https://portal.azure.com/).
2. Select **+ New** on the left of the portal.
3. Search for "Data Science Virtual Machine for Linux (Ubuntu)" in the marketplace.
4. Click *Create* to create an Ubuntu DSVM.
5. Fill in the *Basics* form with the required information. For *Authentication type* select *Password* and take note of your user name and password.
6.  When selecting the location for your VM, note that GPU VMs are only available in certain Azure regions, for example, South Central US. See compute products available by region. Click OK to save the *Basics* information.
6. Choose the size of the virtual machine. Select one of the sizes with NC-prefixed VMs, which are equipped with NVidia GPU chips. Click *View All* to see the full list as needed. 
7. Finish the remaining settings and review the purchase information. Click *Purchase* to create the VM. Take note of the IP address allocated to the virtual machine. You can continue with the next steps while your DSVM is created.

In the next section, you will be creating a compute context as remote docker on the Ubuntu-based Linux DSVM you created in the earlier step. This allows an easy path to execution on a remote DSVM with GPUs.

## Create compute target as remote docker
1. From Azure ML Workbench, launch the Command Line Window by clicking **Open Command Prompt** under **File** menu. 
2. Enter the following command by providing your own values for IP address, username, and password for the Virtual Machine(VM) you created in the earlier section. 

```
az ml computetarget attach remotedocker --name myvm --address [Your VM's IP Address Here!!] --username [Your User Name Here!!] --password 
[Your Password Here!!] 
```

## Configure Azure ML Workbench to Access GPU
1. Go back to Azure ML Workbench and click on the **Files** which is the last item under the home icon on the left bar.
2. Under **Search Files** expand the **aml_config** folder. You will see two new files **myvm.compute** and **myvm.runconfig** (if not , hit the **Refresh** button).
3. Open myvm.compute and change the **baseDockerImage** to **microsoft/mmlspark:plus-gpu-0.9.9** and add a new line **nvidiaDocker: true** and save the file. The file should have the following two lines:
```
baseDockerImage: microsoft/mmlspark:plus-gpu-0.9.9
nvidiaDocker: true
```
4. Now, open **myvm.runconfig** and change **Framework** to **Python** and save the file. The file should have the following line:
```
Framework: Python
```
5. Next, run the following command on the Command Line Window  to prepare the Docker image on your DSVM.

```
az ml experiment prepare -c myvm
```
# Prepare Training Data

This scenario assumes that you have two classes of images that you would like to classify as fail or pass. You can choose to bring in your own dataset given that you have two categories of images or use a generic dataset from Kaggle. If you are bringing your own dataset, zip your images in a file called train.zip. If not using your own dataset, go to the [Kaggle dataset page](https://www.kaggle.com/c/dogs-vs-cats/data) and click download button for the train.zip file. You will be asked to log in to Kaggle and once logged in you will be redirected back to the dataset page. The size of this train.zip file is 543.16 MB. After downloading the zip file, follow the steps below to upload it to Azure Blob Storage.

## Create a storage account

1. Sign in to the [Azure portal](https://portal.azure.com/).
2. In the Azure portal, expand the menu on the left side to open the menu of services, and choose **More Services**. Then, scroll down to **Storage**, and choose **Storage accounts**. On the Storage Accounts window that appears, choose **+Add**.
3. Enter a name for your storage account which must be between 3 and 24 characters in length and may contain numbers and lowercase letters only. Your storage account name must be unique within Azure. The Azure portal will indicate if the storage account name you select is already in use. You will use this name when executing the data ingestion notebook.
4. Select the subscription in which you want to create the new storage account and specify a new resource group or select an existing resource group.
5. Click **Create** to create the storage account

Next, you will create a container and upload your data.

## Create a container and upload your data

1. Once the storage account is created, click **Blobs** on the the storage account page and choose **+Container**. 
2. In the name field, enter **images** and choose **OK**. 
3. Choose the newly created container images and click **Upload**. Select the location of train.zip file from your computer and click **Upload**. 

Next, you will locate your storage account key which will also be used when executing data ingestion notebook.

## Locate your storage account name and key

1. On the storage account page, click on **Access keys** under Settings. 
2. You will see the name of your storage account and two keys.
3. You will use the value for key1 when executing the data ingestion notebook. You can copy the key using the copy button on the right side when needed.

# Start Notebook Server from command line
To run the notebooks of this scenario, you will need to start the Azure Machine Learning Workbench notebook server. You can either run the notebooks from within the Workbench or you can use your browser. Follow the steps below to start the notebook server from command line (CLI).
1. From Azure ML Workbench, launch the Command Line Window by clicking **Open Command Prompt** under **File** menu.
2. Run the following command.
```
az ml notebook start
```
3. You will leave this window open during the execution of your notebooks as closing it will cause the notebook server to stop. You can monitor this window for notebook related messages.
4. Your default browser will  automatically be launched with Jupyter server pointing to the project home directory. You can also use the URL and token displayed in the CLI window to launch notebooks on other browser windows.
5. Next, click on the first notebook, if you receive a prompt asking for which kernel to use, select the kernel with the name **"Your project name myvm"** with the exception of last notebook which will use the kernel **"Your project name local"**. You will find a note at the beggining of each notebook highlighting which kernel to use for that notebook.
7. Follow the instructions and excute the notebook cells.

created by a Microsoft employee.