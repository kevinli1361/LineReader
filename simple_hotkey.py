from pynput import keyboard

# A set that stores pressed keys
keys_pressed = set()



# When hotkey (ctrl + alt + t) is pressed...
def hotkey_train_activate():
    print('<ctrl>+<alt>+t pressed')

# When hotkey (ctrl + alt + r) is pressed...
def hotkey_record_activate():
    print('<ctrl>+<alt>+r pressed')

# When hotkey (ctrl + alt + p) is pressed...
def hotkey_pause_activate():
    print('<ctrl>+<alt>+p pressed')

def esc_activate():
    print('escape is pressed')
    return False

with keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+t': hotkey_train_activate,
        '<ctrl>+<alt>+r': hotkey_record_activate,
        '<ctrl>+<alt>+p': hotkey_pause_activate,
        '<esc>': esc_activate}) as kg:
    kg.join()






"""
def on_press(key):
    try:
        print(f"Key {key.char} pressed")
        keys_pressed.add(key)
    except AttributeError:
        print(f"Key {key} pressed")
        keys_pressed.add(key)
    print(f"Keys pressed: {keys_pressed}")

    if HOTKEY_TRAIN.issubset(keys_pressed):
        print("!!!!!Training Hotkey Pressed!!!!!")

def on_release(key):
    print(f"{key} released")
    keys_pressed.remove(key)
    
    # if "Escape" is pressed, stop the listener
    if key == keyboard.Key.esc:
        return False
    print(f"Keys pressed: {keys_pressed}")

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
"""


