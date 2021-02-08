import numpy as np
import time
import serial, sys, time, glob, struct

def receive(ser):
    # return 6 bytes from the receive buffer
    # there must be 6 bytes to receive (no error checking)
    r = [0,0,0,0,0,0]
    for i in range (6):
        r[i] = ord(ser.read(1)) #read up to 1 byte 
    reply = r
    replyData = (256**3*reply[5]) + (256**2*reply[4]) + (256*reply[3]) + (reply[2])
    print(replyData,r)
    return replyData, r


def send(ser,device, command, data=0):
    # send a packet using the specified device number, command number, and data
    # The data argument is optional and defaults to zero
    packet = struct.pack('<BBl', device, command, data)# l is long,  L was unsigned long
    # Return a string containing the values v1, v2, ... packed according to the given format. 
    # The arguments must match the values required by the format exactly.
    ser.write(packet)
    # Writes binary data to the serial port. This data is sent as a byte or series of bytes; 
    # sends the characters representing the digits of a number use the print() function instead. 


class zaber(object):
    """https://www.zaber.com/manuals/T-LSM#m-7-240-return-current-position-cmd-60"""
    def __init__(self,tty=None):
        #tty = '/dev/ttyUSB1'
        if tty is None:
            tty = glob.glob('/dev/ttyUSB*')[0]
        
        self.ser = serial.Serial(tty, 9600, 8, 'N', 1, timeout=.5)
        self.timeout=.5
        self.scale = [1,1,1]
        #send(self.ser,0,53,37)
        #receive(self.ser) # get microstepping axis 3
        
    def moveto(self,x,axis=1):
        send(self.ser,axis,20,int(x*self.scale[axis]));
        #time.sleep(self.timeout)
        #receive(ser)
        #ser.flushOutput();ser.flushInput()

    def step(self,x,axis=1):
        send(self.ser,axis,21,int(x*self.scale[axis]));

    def restoreSettings(self,axis=1):
        send(self.ser,axis,36,0)
        
    def home(self,axis=1):
        send(self.ser,axis,1)
        
    def reset(self,axis=1):
        send(self.ser,axis,0,0)

    def renumber(self,axis=0):
        """ change axis numbering (hardcoded) automatic""" 
        send(self.ser,0,2,0)

    def re_renumber(self,number,axis=1):
        """ change axis numbering manual"""         
        send(self.ser,axis,2,number)
        
    #def step(self,axis=1):
    #    send(self.ser,axis,21)

    def stop(self,axis=1):
        send(self.ser,axis,23,0)

    def stopAll(self):
        send(self.ser,0,23,0) # TODO TEST it

    def set_home_speed(self,x,axis=1):
        send(self.ser,axis,41,int(x))
        
    def set_speed(self,x,axis=1):
        """ up to max 10000, load/current dependent"""
        send(self.ser,axis,42,int(x))

    def set_acc(self,x,axis=1):
        send(self.ser,axis,43,int(x))
        
    def set_current(self,x,axis=1):
        """If your application does not require high torque, it is best
            to decrease the driving current to reduce power consumption,
            vibration, and motor heating. Trial and error should suggest an
            appropriate setting. If higher torque is required, it is
            generally safe to overdrive motors as long as they are not
            operated continuously. Motor temperature is typically the best
            indication of the degree to which overdriving can be
            employed. If the motor gets too hot to touch (>75°C), you should
            reduce the running current.

        The current is related to the data by the formula:

            Current = CurrentCapacity * 10 / CommandData

        The range of accepted values is 0 (no current), 10 (max) - 127 (min). CurrentCapacity is the hardware's maximum capability of output current.

        To prevent damage, some devices limit the maximum output current to a lower value. In that case the valid range is 0, Limit - 127. Current limits are listed under the device specifications.

        Some devices limit the voltage rather than the current. In this case the same formula can be used by replacing Current and CurrentCapacity with Voltage and PowerSupplyVoltage.

        For example, Suppose you connect a stepper motor rated for 420mA per phase to a T-CD2500. Reversing the equation above and using 420mA as Current gives:

        CommandData (=x here)
            = 10 * CurrentCapacity / Current
            = 10 * 2500mA / 420mA
            = 59.5 (round to 60)

        Therefore CommandData = x =  60.
        """
        send(self.ser,axis,38,int(x));

    def set_hold_current(self,x,axis=1):
        send(self.ser,axis,39,int(x))
        
    def get_id(self,axis=1):
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,50,0)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep)
        return rep

    def get_position(self,axis=1):
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,60,0)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,'microsteps')
        self.x =rep
        return rep

    def get_speed(self,axis=1):
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,53,42)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,'microsteps/p')
        self.x =rep
        return rep

    def get_home_speed(self,axis=1):
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,53,41)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,'microsteps/p')
        self.x =rep
        return rep
    
    
    def get_acc(self,axis=1):
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,53,43)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,'microsteps/per')
        self.x = rep
        return rep


    def get_current(self,axis=1):
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis, 53,38)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,' dn')
        self.x = rep
        return rep

    def get_microsteps(self,axis=1):
        self.ser.flushOutput()
        self.ser.flushInput()        
        send(self.ser,axis, 53, 37)        
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,'N u-steps ')
        self.x = rep
        return rep
    

    def get_hold_current(self,axis=1):
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,53,39)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,'dn')
        self.x = rep
        return rep

    def set_device_mode(self,x,axis=1):
        """
        see modes: https://www.zaber.com/manuals/T-LSM#m-7-119-set-device-mode-cmd-40
        anti-backlash anti-stiction scenarios
        # untested def is x=2176 
        """
        #s='0'*15
        #x = int('0000000000000')
        send(self.ser,axis,40,x)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,'dn')
        self.x = rep
        return rep
    
    def get_device_mode(self,axis=1):
        """
        see modes: https://www.zaber.com/manuals/T-LSM#m-7-119-set-device-mode-cmd-40
        anti-backlash anti-stiction scenarios
        """
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,53,40)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print(rep,'dn')
        self.x = rep
        return rep

    
    def wait(self,axis=1):
        """ wait till motor stops"""
        while(True):
            s = self.status(axis)
            if s == 0:
                break
            else:
                time.sleep(.2)

    def status(self,axis=1):
        """
        Possible status codes are as follows:

        0 - idle, not currently executing any instructions
        1 - executing a home instruction
        10 - executing a manual move (i.e. the manual control knob is turned)
        18 - executing a move to stored position instruction (FW 5.04 and up only)
        20 - executing a move absolute instruction
        21 - executing a move relative instruction
        22 - executing a move at constant speed instruction
        23 - executing a stop instruction (i.e. decelerating)
        """

        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,54,1)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print('status:', rep)
        self.x = rep
        return rep

    def echo(self,data=111,axis=1):
        """
        com test
        """
        self.ser.flushOutput()
        self.ser.flushInput()
        send(self.ser,axis,55,data)
        time.sleep(.25)
        rep,_ = receive(self.ser)
        print('axis:',axis,' received:',rep)
        self.x = rep
        return rep
    def close(self):
        self.ser.close()

def test1(tty=None):
        w=zaber(tty)
        w.restoreSettings()
        time.sleep(1)
        w.home()
        print('wait...')        
        w.wait()
        
        w.step(64*200*10)# 20 rotations 64 usteps, 200steps per rev
        print('wait...')        
        w.wait()
        print('...done')
        w.step(-64*200*10)
        w.wait()
        w.home()
        print('wait...')        
        w.wait()
        
        w.step(64*200*35)
        w.set_speed(100)
        time.sleep(.5)
        w.set_speed(300)
        time.sleep(.5)
        w.set_speed(1000)
        time.sleep(.3)
        w.set_speed(5000)
        time.sleep(.3)
        
        w.close()

        
# Quick Command Reference

# The following table offers a quick command reference for motorized devices running firmware version 5xx. For convenience, you may sort the table below by instruction name, command number, or reply number. Follow the links to view a detailed description of each instruction.
# Instruction Name 	Command# 	Command Data 	Command Type 	Reply Data
# Reset 	0 	Ignored 	Command 	None
# Home 	1 	Ignored 	Command 	Final position (in this case 0)
# Renumber* 	2 	Ignored 	Command 	Device ID
# Move Tracking 	8 	n/a 	Reply 	Tracking Position
# Limit Active 	9 	n/a 	Reply 	Final Position
# Manual Move Tracking 	10 	n/a 	Reply 	Tracking Position
# Store Current Position* 	16 	Address 	Command 	Address
# Return Stored Position 	17 	Address 	Command 	Stored Position
# Move To Stored Position 	18 	Address 	Command 	Final Position
# Move Absolute 	20 	Absolute Position 	Command 	Final Position
# Move Relative 	21 	Relative Position 	Command 	Final Position
# Move At Constant Speed 	22 	Speed 	Command 	Speed
# Stop 	23 	Ignored 	Command 	Final Position
# Read Or Write Memory* 	35 	Data 	Command 	Data
# Restore Settings* 	36 	Peripheral ID 	Command 	Peripheral ID
# Set Microstep Resolution* 	37 	Microsteps 	Setting 	Microsteps
# Set Running Current* 	38 	Value 	Setting 	Value
# Set Hold Current* 	39 	Value 	Setting 	Value
# Set Device Mode* 	40 	Mode 	Setting 	Mode
# Set Home Speed* 	41 	Speed 	Setting 	Speed
# Set Target Speed* 	42 	Speed 	Setting 	Speed
# Set Acceleration* 	43 	Acceleration 	Setting 	Acceleration
# Set Maximum Position* 	44 	Range 	Setting 	Range
# Set Current Position 	45 	New Position 	Setting 	New Position
# Set Maximum Relative Move* 	46 	Range 	Setting 	Range
# Set Home Offset* 	47 	Offset 	Setting 	Offset
# Set Alias Number* 	48 	Alias Number 	Setting 	Alias Number
# Set Lock State* 	49 	Lock Status 	Command 	Lock Status
# Return Device ID 	50 	Ignored 	Read-Only Setting 	Device ID
# Return Firmware Version 	51 	Ignored 	Read-Only Setting 	Version
# Return Power Supply Voltage 	52 	Ignored 	Read-Only Setting 	Voltage
# Return Setting 	53 	Setting Number 	Command 	Setting Value
# Return Status 	54 	Ignored 	Read-Only Setting 	Status
# Echo Data 	55 	Data 	Command 	Data
# Return Current Position 	60 	Ignored 	Read-Only Setting 	Position
# Error 	255 	n/a 	Reply 	Error Code

# * The settings for these commands are saved in non-volatile memory, i.e. the setting persists even if the device is powered down. To restore all settings to factory default, use command 36. 



# All T-Series devices use the same RS232 communications protocol. Your communications settings must be: 9600 baud, no hand shaking, 8 data bits, no parity, one stop bit. The yellow LED will light when there is activity on the RS232 lines. You may use this feature to determine which COM port you are connected to. We recommend using the Zaber Console that you can download from our web site. The source code is also available for you to use as an example for writing your own custom code. See the troubleshooting section later in this manual if you have trouble communicating with the device.

# Important: The first time you connect a device to your computer you must issue a renumber instruction to assign each device a unique identifier. This should be done after all the devices in the daisy-chain are powered up. In older firmware versions (prior to version 5xx) you must issue a renumber instruction after each powerup. In firmware 5xx and up, the device number is stored in non-volatile memory and will persist after powerdown, so you need only issue the renumber instruction when you add new devices to the chain, or rearrange the order of the devices, however it does no harm to issue the renumber instruction after every powerup. You must not transmit any instructions while the chain is renumbering or the renumbering routine may be corrupted. Renumbering takes less than a second, after which you may start issuing instructions over the RS232 connection.

# All instructions consist of a group of 6 bytes. They must be transmitted with less than 10 ms between each byte. If the device has received less than 6 bytes and then a period longer than 10 ms passes, it ignores the bytes already received. We recommend that your software behave similarly when receiving data from the devices, especially in a noisy environment like a pulsed laser lab.

# The following table shows the instruction format:
        
#     All devices renumber: 0, 2, 0, 0, 0, 0
#     All devices home: 0, 1, 0, 0, 0, 0
#     All devices return firmware version: 0, 51, 0, 0, 0, 0
#     Device 1 move to an absolute position (command 20) of 257 microsteps: 1, 20, 1, 1, 0, 0
#     Device 2 move to a relative position (command 21) of -1 microstep: 2, 21, 255, 255, 255, 255

# Most instructions cause the device to reply with a return code. It is also a group of 6 bytes. The first byte is the device #. Byte #2 is the instruction just completed or 255 (0xFF) if an error occurs. Bytes 3, 4, 5 and 6 are data bytes in the same format as the instruction command data.
# Data Conversion Algorithms

# If you are writing software to control Zaber products, you'll likely need to generate data bytes 3 through 6 from a single data value, or vise versa. The following pseudo-code can be used as a model.

# Converting command data into command bytes to send to Zaber products

# If Cmd_Data < 0 then Cmd_Data = 256^4 + Cmd_Data                                        'Handles negative data

# Cmd_Byte_6 = Cmd_Data / 256^3

# Cmd_Data   = Cmd_Data - 256^3 * Cmd_Byte_6

# Cmd_Byte_5 = Cmd_Data / 256^2

# Cmd_Data   = Cmd_Data - 256^2 * Cmd_Byte_5

# Cmd_Byte_4 = Cmd_Data / 256

# Cmd_Data   = Cmd_Data - 256   * Cmd_Byte_4

# Cmd_Byte 3 = Cmd_Data

# Converting reply bytes into a single reply data value

# Reply_Data = 256^3 * Rpl_Byte 6 + 256^2 * Rpl_Byte_5 + 256 * Rpl_Byte_4 + Rpl_Byte_3

# If Rpl_Byte_6 > 127 then Reply_Data = Reply_Data - 256^4                                'Handles negative data

