# MonsterVision

## IMPORTANT NOTE:
VSCode supports Github, but you may need to update your Raspberry Pi:

  sudo apt-get install gnome-keyring

### To sign in:

1. Before launching VSCode, open a browser and go to http://github.com.
2. Sign into Github using **your** credentials.
3. Launch VSCode.  Wait for all extensions to load (watch the bottom right corner of the window).
4. Once the extensions have all loaded, click on the "user" icon at the left edge of the window, close to the bottom.  It will probably have a #1 in a circle on it.
5. A series of browser windows and dialog boxes will appear.  Follow the instructions given to log in.
6. **Important**: You can ignore the error message box that appears that says "Writing login information to the keychain...".  I haven't been able to fix it.

### To Sign Out

1. Click on the "user" icon (as above) and select __your name__ -> Sign Out.
2. Exit VSCode.
3. Shut down the browser.


## Directory Structure

### resources/nn
Each directory is one NN model.

### Top Level Files
cone_tracker.py   A very basic OAK program that uses the trafficcones NN to find and localize traffic cones.
