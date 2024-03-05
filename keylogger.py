from pynput import keyboard
import os
def keyPressed(key):
    
    with open('keylo.txt','a+') as lof:
        try:
            char = key.char
            if char=='Key.space':
                char=' '
            lof.write(char)
            print(char)
        except:
            lof.write("\t")
            print("Errror")

lister=keyboard.Listener(on_press=keyPressed)
lister.start()
input()
file_path = "D:\\kongu\\Downloads\\DepressionDetection\\keylo.txt"
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File '{file_path}' deleted successfully.")
else:
    print(f"File '{file_path}' does not exist.")