import usb.core
dev = usb.core.find(idVendor=0x125F, idProduct=0x312B)

print(dev)