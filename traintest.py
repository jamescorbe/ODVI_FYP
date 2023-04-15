from ultralytics import YOLO, checks, hub

def train():
    checks()
    hub.login('47f571b59c7b2052c309b2f8bb6a6f46bf787cff1c')

    model = YOLO('https://hub.ultralytics.com/models/0gl7we1buRRhfg2gjeQw')
    model.train()

if __name__ == '__main__':
     train()