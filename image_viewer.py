
import os
import wx
from wx.lib.pubsub import pub
import csv
import shutil
from tempfile import NamedTemporaryFile

########################################################################
class ViewerPanel(wx.Panel):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        
        width, height = wx.DisplaySize()

        self.class_labels = {0: 'background', 1: 'buddy', 2: 'jade', 3: 'lucy', 4: 'tim'}
        self.class_labels_rev = {'background': 0, 'buddy':1, 'jade':2, 'lucy':3, 'tim':4}

        self.csvFields = ['frame','xmin','xmax','ymin','ymax','class_id']
        self.tempCsvFile = NamedTemporaryFile(mode='w', delete=False, newline='')

        self.startingRow = 0
        self.rowNumber = 0

        # 20180706_215_cats_807,138,231,94,194,1
        self.frame = ''
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.classification = 0
        self.newClassification = 0

        self.currentPicture = 0
        self.totalPictures = 0
        self.photoMaxSize = 300
        pub.subscribe(self.loadTrainingCSV, ("update images"))

        self.slideTimer = wx.Timer(None)
        self.slideTimer.Bind(wx.EVT_TIMER, self.update)

        self.layout()

    #----------------------------------------------------------------------
    def layout(self):
        """
        Layout the widgets on the panel
        """

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)

        img = wx.Image(self.photoMaxSize,self.photoMaxSize)
        self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY,
                                         wx.Bitmap(img))
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL|wx.CENTER, 5)
        self.imageLabel = wx.StaticText(self, label="")
        self.mainSizer.Add(self.imageLabel, 0, wx.ALL|wx.CENTER, 5)

        btnData = [("Next", btnSizer, self.onNext),
                   ("Update CSV", btnSizer, self.onUpdateCSV),
                   ("Done", btnSizer, self.onDone)]
        for data in btnData:
            label, sizer, handler = data
            self.btnBuilder(label, sizer, handler)

        self.mainSizer.Add(btnSizer, 0, wx.CENTER)
        self.SetSizer(self.mainSizer)

    #----------------------------------------------------------------------
    def btnBuilder(self, label, sizer, handler):
        """
        Builds a button, binds it to an event handler and adds it to a sizer
        """
        btn = wx.Button(self, label=label)
        btn.Bind(wx.EVT_BUTTON, handler)
        sizer.Add(btn, 0, wx.ALL|wx.CENTER, 5)

    #----------------------------------------------------------------------
    def loadImage(self, image):
        """"""
        img = wx.Image(image, wx.BITMAP_TYPE_ANY)

        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
            NewW = self.photoMaxSize
            NewH = self.photoMaxSize * H / W
        else:
            NewH = self.photoMaxSize
            NewW = self.photoMaxSize * W / H
        img = img.Scale(NewW,NewH)

        imgBit = wx.BitmapFromImage(img)
        dc = wx.MemoryDC(imgBit)
        dc.SetPen(wx.Pen(wx.RED, 1))
        text = self.class_labels[self.classification]
        tw, th = dc.GetTextExtent(text)
        dc.SetTextForeground(wx.RED)
        dc.DrawText(text, (self.xmax-tw), (self.ymax-th))
        dc.SetBrush(wx.Brush(wx.RED, wx.TRANSPARENT)) #set brush transparent for non-filled rectangle
        dc.DrawRectangle(self.xmin,self.ymin,self.xmax-self.xmin,self.ymax-self.ymin)
        dc.SelectObject(wx.NullBitmap)

        self.imageCtrl.SetBitmap(imgBit)
        label = "Row:{:04d}   Image: {}   xmin:{:03d}   xmax:{:03d}   ymin:{:03d}   ymax:{:03d}   class: {}".format(self.rowNumber, self.frame, self.xmin, self.xmax, self.ymin, self.ymax, self.class_labels[self.classification])
        self.imageLabel.SetLabel(label)
        self.Refresh()
        pub.sendMessage(topicName="resize", msg="")

    #----------------------------------------------------------------------
    def nextPicture(self):
        """
        Loads the next picture in the directory
        """
        self.csvwriter.writerow(self.line)
        self.rowNumber += 1

        self.line = next(self.csvreader)
        self.setImageParameters(self.line)
        self.currentImage = self.imagePath + '\\' + self.frame

        self.loadImage(self.currentImage)

    #----------------------------------------------------------------------
    def update(self, event):
        """
        Called when the slideTimer's timer event fires. Loads the next
        picture from the folder by calling th nextPicture method
        """
        self.nextPicture()

    #----------------------------------------------------------------------
    def loadTrainingCSV(self, msg):
        self.imagePath = msg;
        self.csvFile = open(self.imagePath+'\\train.csv', 'r')

        self.csvreader = csv.DictReader(self.csvFile, fieldnames=self.csvFields)
        self.csvwriter = csv.DictWriter(self.tempCsvFile, fieldnames=self.csvFields)

        for x in range(0,self.startingRow):
            self.line = next(self.csvreader)
            self.csvwriter.writerow(self.line)
            self.rowNumber += 1

        self.line = next(self.csvreader)
        self.setImageParameters(self.line)
        self.currentImage = self.imagePath + '\\' + self.frame
        self.loadImage(self.currentImage)

    #----------------------------------------------------------------------
    def setImageParameters(self, line):
        self.frame = line[self.csvFields[0]]
        self.xmin = int(line[self.csvFields[1]])
        self.xmax = int(line[self.csvFields[2]])
        self.ymin = int(line[self.csvFields[3]])
        self.ymax = int(line[self.csvFields[4]])
        self.classification = int(line[self.csvFields[5]])

    #----------------------------------------------------------------------
    def onNext(self, event):
        """
        Calls the nextPicture method
        """
        self.nextPicture()

    #----------------------------------------------------------------------
    def onUpdateCSV(self, event):
        """
        Opens a DirDialog to allow the user to open a folder with pictures
        """
        dlg = MyDialog()

        if dlg.ShowModal() == wx.ID_OK:
            self.newClassification = self.class_labels_rev[dlg.comboBox1.GetValue()]
            self.line[self.csvFields[5]] = self.newClassification
            print (self.newClassification)
            self.setImageParameters(self.line)
            self.loadImage(self.currentImage)

    def onDone(self, event):
        self.csvwriter.writerow(self.line)
        for row in self.csvreader:
            self.csvwriter.writerow(row)
        self.csvFile.close()
        self.tempCsvFile.close()

        shutil.move(self.tempCsvFile.name, self.csvFile.name)

        quit()

########################################################################
class MyDialog(wx.Dialog):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Dialog.__init__(self, None, title="Update classification")

        self.comboBox1 = wx.ComboBox(self,
                                     choices=['background', 'buddy', 'jade', 'lucy', 'tim'],
                                     value="")
        okBtn = wx.Button(self, wx.ID_OK)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.comboBox1, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(okBtn, 0, wx.ALL|wx.CENTER, 5)
        self.SetSizer(sizer)

########################################################################
class ViewerFrame(wx.Frame):
    """"""

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Frame.__init__(self, None, title="Image Viewer")
        panel = ViewerPanel(self)
        self.folderPath = ""
        pub.subscribe(self.resizeFrame, ("resize"))

        self.initToolbar()
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(self.sizer)

        self.Show()
        self.sizer.Fit(self)
        self.Center()


    #----------------------------------------------------------------------
    def initToolbar(self):
        """
        Initialize the toolbar
        """
        self.toolbar = self.CreateToolBar()
        self.toolbar.SetToolBitmapSize((16,16))

        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        openTool = self.toolbar.AddSimpleTool(wx.ID_ANY, open_ico, "Open", "Open an Image Directory")
        self.Bind(wx.EVT_MENU, self.onOpenDirectory, openTool)

        self.toolbar.Realize()

    #----------------------------------------------------------------------
    def onOpenDirectory(self, event):
        """
        Opens a DirDialog to allow the user to open a folder with pictures
        """
        dlg = wx.DirDialog(self, "Choose a training csv",
                           style=wx.DD_DEFAULT_STYLE)

        if dlg.ShowModal() == wx.ID_OK:
            self.folderPath = dlg.GetPath()
            print (self.folderPath)
        pub.sendMessage(topicName="update images", msg=self.folderPath)
        
    #----------------------------------------------------------------------
    def resizeFrame(self, msg):
        """"""
        self.sizer.Fit(self)
        
#----------------------------------------------------------------------
if __name__ == "__main__":
    app = wx.App()
    frame = ViewerFrame()
    app.MainLoop()
    