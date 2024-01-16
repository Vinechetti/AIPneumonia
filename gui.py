import tkinter as Tk
import customtkinter as CTk
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets
from PIL import Image
from tkinter import messagebox


global detect_flag, first_time
detect_flag = False
first_time = True


def main():
    root = CTk.CTk()
    root.geometry("800x720")
    root.resizable(False, False)
    root.title("Pneumonia AI detector")

    welcome = CTk.CTkLabel(
        root, text="Welcome to Pneumonia AI detector! Please select an image to test..."
    )
    welcome.pack()

    image_display = CTk.CTkLabel(root, text="")
    open_file = CTk.CTkButton(
        root, text="Select a file...", command=lambda: open_image(root, image_display)
    )
    open_file.pack(pady=10)
    image_display.pack(pady=10)

    recalculate = messagebox.askyesno(
        "Recalculate accuracy?",
        "Do you want to recalculate accuracy? Warning: It will take a while.",
    )
    if recalculate:
        messagebox.showinfo(
            "Recalculate accuracy?",
            "The program will now recalculate the accuracy. Check the console for current progress.",
        )
        acc = recalculate_accuracy()
    root.mainloop()


class pneumonia(nn.Module):
    def __init__(self, num_classes=2):
        super(pneumonia, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1
        )

        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32 * 112 * 112, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 112 * 112)
        output = self.fc(output)

        return output


def recalculate_accuracy():
    try:
        model = torch.load(model_file)
        model.eval()
    except FileNotFoundError:
        messagebox.showerror(
            "Model file not found!",
            "Model file was not found. Please specify a correct path in the config file.",
        )
        exit()
    cwd = os.getcwd()
    try:
        test_dir = os.path.abspath(".\\pneumonia\\chest_xray\\test")
        data_T = T.Compose(
            [
                T.Resize(size=(224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        testset = datasets.ImageFolder(test_dir, transform=data_T)
        testloader = DataLoader(testset, batch_size=64, shuffle=True)
        correct_count, all_count, progress = 0, 0, 0
        for images, labels in testloader:
            for i in range(len(labels)):
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                img = images[i].view(1, 3, 224, 224)
                with torch.no_grad():
                    logps = model(img)
                    ps = torch.exp(logps)
                    probab = list(ps.cpu()[0])
                    pred_label = probab.index(max(probab))
                    true_label = labels.cpu()[i]
                    if true_label == pred_label:
                        correct_count += 1
                    all_count += 1
                progress += 1
            print(f"Recalculating accuracy: {progress}/{len(testloader)*64}")
        print(f"Calculated accuracy: {correct_count/all_count}")
        global acc
        acc = correct_count / all_count
        with open("config.cfg", "r") as f:
            lines = f.readlines()
            lines[0] = f"acc={acc}\n"
        with open("config.cfg", "w") as file:
            file.writelines(lines)
        return correct_count / all_count
    except WindowsError:
        relocate = messagebox.askyesno(
            "Test images set not found!",
            "Test images set was not found, which is necessary for recalculating accuracy. Do you want to locate it?",
        )
        if relocate:
            test_dir = CTk.filedialog.askdirectory()
            data_T = T.Compose(
                [
                    T.Resize(size=(224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            testset = datasets.ImageFolder(test_dir, transform=data_T)
            testloader = DataLoader(testset, batch_size=64, shuffle=True)
            correct_count, all_count, progress = 0, 0, 0
            for images, labels in testloader:
                for i in range(len(labels)):
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()
                    img = images[i].view(1, 3, 224, 224)
                    with torch.no_grad():
                        logps = model(img)
                        ps = torch.exp(logps)
                        probab = list(ps.cpu()[0])
                        pred_label = probab.index(max(probab))
                        true_label = labels.cpu()[i]
                        if true_label == pred_label:
                            correct_count += 1
                        all_count += 1
                    progress += 1
                print(f"Recalculating accuracy: {progress}/{len(testloader)*64}")
            acc = correct_count / all_count
            with open("config.cfg", "r") as f:
                lines = f.readlines()
                lines[0] = f"acc={acc}\n"
            with open("config.cfg", "w") as file:
                file.writelines(lines)
            print(f"Calculated accuracy: {correct_count/all_count}")
        else:
            with open("config.cfg", "r") as f:
                lines = f.readlines()
                lines[0] = f"acc=0.8221\n"
            with open("config.cfg", "w") as file:
                file.writelines(lines)
            return 0.8221
    except Exception as e:
        print(e)
        messagebox.showwarning(
            "Failed!",
            "Recalculating accuracy failed. Program will use the default accuracy calculated beforehand.",
        )
        return 0.8221


with open("config.cfg", "r") as f:
    cfg = []
    for line in f.readlines():
        if line[0] != "#":
            cfg.append(line.split("=")[1].strip("\n").strip('"'))
    try:
        global acc
        acc = float(cfg[0])
    except ValueError:
        model_file = cfg[1]
        messagebox.showwarning(
            "Config file corrupted",
            "Config file is corrupted. You'll have to recalculate accuracy. You can track progress in the console.",
        )
        recalculate_accuracy()
    model_file = cfg[1]


def open_image(root, image_display):
    global file_path
    file_path = Tk.filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )

    if file_path:
        image_path = file_path

        # Resize and display the image
        global img
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((640, 480), Image.LANCZOS)
        img_resized = CTk.CTkImage(img_resized, size=(640, 480))

        # Update the image display
        image_display.configure(image=img_resized)
        image_display.image = img_resized
        global detect_flag
        if not detect_flag:
            detect = CTk.CTkButton(
                root, text="Detection", command=lambda: detection(root)
            )
            detect.pack(pady=10)
            detect_flag = True
        global first_time, output_label
        if not first_time:
            output_label.configure(text="", fg_color="transparent")


def detection(root):
    try:
        model = torch.load(model_file)
        model.eval()
    except FileNotFoundError:
        messagebox.showerror(
            "Model file not found!",
            "Model file was not found. Please make sure it's in the same directory as the application. Make sure its name is the same one as specified in config file.",
        )
        exit()
    data_T = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    global img
    img_tensor = data_T(img)
    img_new = img_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        logps = model(img_new)
    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    global first_time
    global output_label
    if first_time:
        if not pred_label:
            output_label = CTk.CTkLabel(
                root,
                text=f"This patient does NOT suffer from pneumonia.\nAccuracy: {round(acc*100, 2)}%",
                fg_color="green",
            )
            output_label.pack(pady=10)
        else:
            output_label = CTk.CTkLabel(
                root,
                text=f"This patient does suffer from pneumonia.\nAccuracy: {round(acc*100, 2)}%",
                fg_color="red",
            )
            output_label.pack(pady=10)
        first_time = False
    else:
        if not pred_label:
            output_label.configure(
                text=f"This patient does NOT suffer from pneumonia.\nAccuracy: {round(acc*100, 2)}%",
                fg_color="green",
            )
        else:
            output_label.configure(
                text=f"This patient does suffer from pneumonia.\nAccuracy: {round(acc*100, 2)}%",
                fg_color="red",
            )


if __name__ == "__main__":
    main()
