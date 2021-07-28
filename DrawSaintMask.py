from PyQt5 import QtWidgets, QtCore, QtGui, uic
import pyqtgraph as pg
import sys, os
import numpy as np
from scipy import ndimage as ndi
from qtrangeslider import QLabeledRangeSlider
from collections import defaultdict
import pickle

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
        # add new detector formats
        # key (string) : file extension
        # value (tuple): name   (for Saint)
        #              : rows   (dim 0)
        #              : cols   (dim 1)
        #              : offset (header size)
        #              : dtype  (datatype)
        #              : rot    (clock-wise by 90 deg, bool)
        #                  key        name        rows  cols offset     dtype    rot
        self.formats = {'*.raw':('CMOS-PHOTONII', 1024,  768,     0, np.int32, False),
                        '*.tif':('PILATUS'      , 1043,  981,  4096, np.int32, True ),}
        
        #self.color_mask = (102, 0, 51, 255)
        self.color_mask = (0, 0, 0, 255)
        self.current_dir = os.getcwd()
        
        self.init_gui()
        self.init_range_slider()
        self.init_image_item()
        self.init_patches()
        self.init_mask_lines()
        self.init_mask_negative()
        self.load_file_dialog()
        
    def init_gui(self):
        uic.loadUi('DrawSaintMask.ui', self)
        
        self.glw.setAspectLocked(True)
        self.plt = self.glw.addPlot()
        self.plt.setAspectLocked()
        self.plt.hideAxis('bottom')
        self.plt.hideAxis('left')
        
        self.action_save.triggered.connect(self.mask_prepare)
        self.action_save_as.triggered.connect(self.save_as_file_dialog)
        self.action_open.triggered.connect(self.load_file_dialog)
        self.action_circs_add.triggered.connect(self.patches_circs_add)
        self.action_circs_rem.triggered.connect(self.patches_circs_rem)
        self.action_mask_lines.toggled.connect(self.mask_lines_toggle)
        self.action_mask_negative.toggled.connect(self.mask_negative_toggle)
    
    def init_parameter(self):
        self.parameter = defaultdict(list)
        self.parameter['img_lvl_min'] = 0
        self.parameter['img_lvl_max'] = None
        self.parameter['flag_mask_negative'] = False
        self.parameter['flag_mask_lines'] = False
        
        self.flag_toggle = {'flag_mask_negative':self.action_mask_negative,
                            'flag_mask_lines':self.action_mask_lines}
        self.flag_toggle['flag_mask_negative'].setChecked(self.parameter['flag_mask_negative'])
        self.flag_toggle['flag_mask_lines'].setChecked(self.parameter['flag_mask_lines'])
    
    def init_range_slider(self):
        '''
        https://pypi.org/project/QtRangeSlider/
        p3 -m pip install qtrangeslider[pyqt5]
        '''
        self.range_slider_img = QLabeledRangeSlider(QtCore.Qt.Horizontal)
        self.range_slider_img.setHandleLabelPosition(2)
        self.range_slider_img.setEdgeLabelMode(1)
        self.range_slider_img.setSingleStep(1)
        self.range_slider_img.setPageStep(1)
        self.range_slider_img.valueChanged.connect(self.image_update_from_slider)
        self.horizontalLayout_bottom.insertWidget(1, self.range_slider_img)
    
    def init_image_item(self):
        pg.setConfigOptions(imageAxisOrder='row-major', background='k', leftButtonPan=True)
        self.img = pg.ImageItem()
        # Monkey-patch the image to use our custom hover function.
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this.
        self.img.hoverEvent = self.imageHoverEvent
        
        self.plt.addItem(self.img)
        colors = [(255, 255, 255),(150, 150, 150),
                  (  0,   0,   0),(255,   0,   0),
                  (255, 150,   0),(255, 255,   0),
                  ( 0, 255, 255)]
        
        cmap = pg.ColorMap(pos=[.0, .02, .05, .1, .5, .95, 1.], color=colors)
        lut = cmap.getLookupTable(nPts=2048)
        lut[:1, :] = np.array([220, 220, 220])
        self.img.setLookupTable(lut)
        self.img.setZValue(-2)
    
    def init_mask_negative(self):
        colors = [self.color_mask,(0, 0, 0, 0)]
        cmap = pg.ColorMap(pos=[0, 1], color=colors)
        lut = cmap.getLookupTable(nPts=2)
        self.neg = pg.ImageItem()
        self.plt.addItem(self.neg)
        self.neg.setLookupTable(lut)
        self.neg.setZValue(-1)
        if self.action_mask_negative.isChecked():
            self.neg.show()
        else:
            self.neg.hide()
    
    def load_mask_negative(self):
        self.neg.setImage(np.sign(self.img.image))
    
    def init_mask_lines(self):
        colors = [self.color_mask,(0, 0, 0, 0)]
        cmap = pg.ColorMap(pos=[0, 1], color=colors)
        lut = cmap.getLookupTable(nPts=2)
        self.lin = pg.ImageItem()
        self.plt.addItem(self.lin)
        self.lin.setLookupTable(lut)
        self.lin.setZValue(-1)
        if self.action_mask_lines.isChecked():
            self.lin.show()
        else:
            self.lin.hide()
    
    def load_mask_lines(self):
        arr = np.ones(self.img.image.shape)
        arr[:,[0, 63, 64, 191, 192, 319, 320, 447, 448, 575, 576, 703, 704, 767]] = 0
        self.lin.setImage(arr)
    
    def init_patches(self):
        self.patches_base = []
        self.patches_circs = []
        self.patch_parameter = {'scaleSnap':False,'translateSnap':False,
                                'rotateSnap':False,'handlePen':pg.mkPen(0, 0, 0, 255),
                                'hoverPen':pg.mkPen((51, 255, 153, 255), width=3),
                                'handleHoverPen':pg.mkPen((51, 255, 153, 255), width=3)}
        
        self.patch_size_default = 100
        self.patch_size_current = self.patch_size_default
        self.patch_size_increment = 50
    
    def mask_negative_toggle(self, toggle):
        if toggle:
            self.parameter['flag_mask_negative'] = True
            self.neg.show()
        else:
            self.parameter['flag_mask_negative'] = False
            self.neg.hide()
    
    def mask_lines_toggle(self, toggle):
        if toggle:
            self.parameter['flag_mask_lines'] = True
            self.lin.show()
        else:
            self.parameter['flag_mask_lines'] = False
            self.lin.hide()
    
    def parameter_set(self, par, val):
        if par in self.parameter:
            self.parameter[par] = val
    
    def parameter_dump(self):
        with open(self.path_mask, 'wb') as wf:
            pickle.dump(self.parameter, wf)
    
    def patches_circs_add(self):
        x = self.img_dim_x/2 - self.patch_size_current/2 - self.patch_size_increment/2
        y = self.img_dim_y/2 - self.patch_size_current/2 - self.patch_size_increment/2
        self.patch_size_current += self.patch_size_increment
        patch_add = pg.CircleROI((x,y), (self.patch_size_current,self.patch_size_current), pen=pg.mkPen((0, 255, 0, 255), width=3), **self.patch_parameter)
        patch_add.sigRegionChangeFinished.connect(self.patches_circs_sort)
        self.plt.addItem(patch_add)
        self.patches_circs.append((patch_add,1))
        
        x = x - self.patch_size_increment/2
        y = y - self.patch_size_increment/2
        self.patch_size_current += self.patch_size_increment
        patch_sub = pg.CircleROI((x,y), (self.patch_size_current,self.patch_size_current), pen=pg.mkPen(self.color_mask, width=3), **self.patch_parameter)
        patch_sub.sigRegionChangeFinished.connect(self.patches_circs_sort)
        self.plt.addItem(patch_sub)
        self.patches_circs.append((patch_sub,0))
        
        self.patches_circs_sort()
    
    def patches_circs_rem(self):
        if self.patches_circs:
            p,_ = self.patches_circs.pop()
            self.plt.removeItem(p)
            p,_ = self.patches_circs.pop()
            self.plt.removeItem(p)
            self.patches_circs_sort()
    
    def patches_circs_sort(self):
        if self.patches_circs:
            newlist = sorted(self.patches_circs, key=lambda x: x[0].size().manhattanLength(), reverse=True)
            for idx, (obj,_) in enumerate(newlist):
                obj.setZValue(idx)
                size = obj.size().manhattanLength()/2
                if size >= self.patch_size_current:
                    self.patch_size_current = size
        else:
            self.patch_size_current = self.patch_size_default
    
    def patches_clear(self):
        while self.patches_base:
            p,_ = self.patches_base.pop()
            self.plt.removeItem(p)
        while self.patches_circs:
            p,_ = self.patches_circs.pop()
            self.plt.removeItem(p)
        self.patches_circs_sort()
    
    def get_paths(self):
        try:
            stem, ext = os.path.splitext(self.path_img)
            root, info = os.path.split(stem)
            _split = info.split('_')
            frame_num = _split.pop()
            run_num = int(_split.pop())
            name = '_'.join(_split)
            self.path_mask = os.path.join(root, '{}.msk'.format(stem))
            self.path_sfrm = os.path.join(root, '{}_xa_{:>02}_0001.sfrm'.format(name, int(run_num)))
            return True
        except (ValueError, IndexError):
            print('Error parsing image name! Expected format: name_run_image.ext (e.g. name_01_0001.raw)')
            return False
    
    def load_parameter(self):
        if os.path.exists(self.path_mask):
            with open(self.path_mask, 'rb') as rf:
                self.parameter = pickle.load(rf)
            # set keys and toggle connected objects
            for key, val in self.parameter.items():
                if key in self.flag_toggle.keys():
                    self.parameter_set(key, val)
                    self.flag_toggle[key].setChecked(val)
        else:
            self.parameter['base'].append(['rect', (self.img_dim_x/2-10, -10), (20, self.img_dim_y/2), 0.0, 0])
            self.parameter['base'].append(['circ', (self.img_dim_x/2-20, self.img_dim_y/2-20), (40, 40), 0.0, 0])
    
    def load_patches(self):
        if 'base' in self.parameter:
            for name, pos, size, angle, msk in self.parameter['base']:
                if name == 'rect':
                    self.rect = pg.RectROI(pos=pos, size=size, angle=angle, pen=pg.mkPen(self.color_mask, width=3), sideScalers=True, **self.patch_parameter)
                    self.rect.addRotateHandle((0.0,1.0), center=(0.5,0.0))
                    self.rect.addRotateHandle((1.0,0.0), center=(0.5,1.0))
                    self.rect.addScaleHandle((0.5,0.0), center=(0.5,1.0))
                    self.rect.addScaleHandle((0.0,0.5), center=(1.0,0.5))
                    self.rect.addScaleHandle((0.0,0.0), center=(1.0,1.0))
                    self.rect.setZValue(99)
                    self.plt.addItem(self.rect)
                    self.patches_base.append((self.rect, msk))
                elif name == 'circ':
                    self.circ = pg.CircleROI(pos=pos, size=size, angle=angle, pen=pg.mkPen(self.color_mask, width=3), **self.patch_parameter)
                    self.circ.setZValue(100)
                    self.plt.addItem(self.circ)
                    self.patches_base.append((self.circ, msk))
        if 'circles' in self.parameter:
            for idx, (name, pos, size, angle, msk) in enumerate(self.parameter['circles']):
                if msk:
                    self.circ = pg.CircleROI(pos=pos, size=size, angle=angle, pen=pg.mkPen((0, 255, 0, 255), width=3), **self.patch_parameter)
                else:
                    self.circ = pg.CircleROI(pos=pos, size=size, angle=angle, pen=pg.mkPen(self.color_mask, width=3), **self.patch_parameter)
                self.circ.setZValue(idx)
                self.plt.addItem(self.circ)
                self.patches_circs.append((self.circ, msk))
    
    def load_file_dialog(self):
        self.path_img, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', self.current_dir, ' '.join(self.formats.keys()), options=QtWidgets.QFileDialog.DontUseNativeDialog)
        
        if not self.path_img:
            return
        
        if not self.get_paths():
            return
        
        self.current_dir = os.path.dirname(self.path_img)
        self.label_title.setText(self.path_img)
        _, ext = os.path.splitext(self.path_img)
        
        self.patches_clear()
        self.init_parameter()
        self.load_image(self.path_img, *self.formats['*' + ext])
        self.load_mask_negative()
        self.load_mask_lines()
        self.load_parameter()
        self.load_patches()
        self.image_update_contrast()
    
    def save_as_file_dialog(self):
        # store current path
        temp_path_sfrm = self.path_sfrm
        # find new path and prepare mask
        self.path_sfrm, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save File', temp_path_sfrm, '.sfrm', options=QtWidgets.QFileDialog.DontUseNativeDialog)
        if self.path_sfrm:
            self.mask_prepare()
        # reset path
        self.path_sfrm = temp_path_sfrm
    
    def load_image(self, img_path, detector_type, rows, cols, offset, datatype, rotate):
        with open(img_path, 'rb') as f:
            f.read(offset)
            data = np.flipud(np.frombuffer(f.read(), datatype).copy().reshape((rows, cols)))
        if rotate:
            data = np.rot90(data)
        self.img.setImage(data)
        self.img_dim_y, self.img_dim_x = data.shape
        self.detector_type = detector_type
    
    def image_update_contrast(self):
        img_nanmax = np.nanmax(self.img.image)
        if self.parameter['img_lvl_max'] is None:
            self.parameter['img_lvl_max'] = img_nanmax/10
        self.img.setLevels([self.parameter['img_lvl_min'], self.parameter['img_lvl_max']])
        self.range_slider_img.setMinimum(-100)
        self.range_slider_img.setMaximum(img_nanmax)
        self.range_slider_img.setValue((self.parameter['img_lvl_min'], self.parameter['img_lvl_max']))
        self.plt.setXRange(0, self.img_dim_x, padding=0)
        self.plt.setYRange(0, self.img_dim_y, padding=0)
    
    def image_update_from_slider(self):
        int_min, int_max = self.range_slider_img.value()
        self.img.setLevels([int_min, int_max])
        self.parameter['img_lvl_min'] = int_min
        self.parameter['img_lvl_max'] = int_max
    
    def mask_add_obj(self, obj, val):
        '''
        Circles and Ellipses
        Note: returnMappedCoords is not yet supported for this ROI type.
        
        Workaround taken from:
        https://groups.google.com/g/pyqtgraph/c/fcysRvIcJi8
        https://groups.google.com/g/pyqtgraph/c/-kNPXxDeERs
        
        Still produces erroneously unmasked regions so we are not going to use ellipses
        The tilting of the rectangle should be only minute, showing only single unmasked pixels
        Application of scipy.ndimage.binary_erosion() before writing the mask should make it smooth and clean.
        - This is bad!
        '''
        cols, rows = self.img.image.shape
        m = np.mgrid[:cols,:rows]
        possx = m[0,:,:]
        possy = m[1,:,:]
        possx.shape = cols, rows
        possy.shape = cols, rows
        mpossx = obj.getArrayRegion(possx, self.img).astype(int)
        mpossy = obj.getArrayRegion(possy, self.img).astype(int)
        self.msk[mpossx, mpossy] = val
    
    def mask_prepare(self):
        self.msk = np.ones(self.img.image.shape)
        
        self.parameter['circles'] = []
        newlist = sorted(self.patches_circs, key=lambda x: x[0].size().manhattanLength(), reverse=True)
        for obj, val in newlist:
            self.mask_add_obj(obj, val)
            self.parameter['circles'].append(['circ', obj.pos(), obj.size(), obj.angle(), val])
        
        self.parameter['base'] = []
        for obj, val in self.patches_base:
            self.mask_add_obj(obj, val)
            if type(obj) == pg.graphicsItems.ROI.RectROI:
                self.parameter['base'].append(['rect', obj.pos(), obj.size(), obj.angle(), val])
            elif type(obj) == pg.graphicsItems.ROI.CircleROI:
                self.parameter['base'].append(['circ', obj.pos(), obj.size(), obj.angle(), val])
        # interpolation fails -> erode the mask
        self.msk = ndi.binary_erosion(self.msk)
        
        # mask negatives?
        if self.parameter['flag_mask_negative']:
            self.msk[self.img.image < 0] = 0
        
        # mask lines?
        if self.parameter['flag_mask_lines']:
            self.msk[self.lin.image == 0] = 0
        
        header = bruker_header()
        # fill known header entries
        header['NCOLS']       = [self.img.image.shape[1]]                 # Number of pixels per row; number of mosaic tiles in X; dZ/dX
        header['NROWS']       = [self.img.image.shape[0]]                 # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
        #header['CCDPARM'][:] = [1.47398, 36.60, 359.8295, 0.0, 163810.0] # readnoise, electronsperadu, electronsperphoton, bruker_bias, bruker_fullscale
        #header['DETTYPE'][:] = ['CMOS-PHOTONII', 37.037037, 1.004, 0, 0.425, 0.035, 1]
        header['DETTYPE'][:]  = [self.detector_type, 10.0, 1.0, 0, 0.0, 0.0, 1] # dettype pix512percm cmtogrid circular brassspacing windowthickness accuratetime
        #header['SITE']       = ['Aarhus Huber Diffractometer']           # Site name
        #header['MODEL']      = ['Microfocus X-ray Source']               # Diffractometer model
        #header['TARGET']     = ['Ag Ka']                                 # X-ray target material)
        #header['SOURCEK']    = [50.0]                                    # X-ray source kV
        #header['SOURCEM']    = [0.880]                                   # Source milliamps
        #header['WAVELEN'][:] = [0.560860, 0.559420, 0.563810]            # Wavelengths (average, a1, a2)
        header['WAVELEN'][:] = [1.0, 1.0, 1.0]                            # Wavelengths (average, a1, a2)
        #header['CORRECT']    = ['INTERNAL, s/n: A110247']                # Flood correction filename
        #header['DARK']       = ['INTERNAL, s/n: A110247']                # Dark current frame name
        #header['WARPFIL']    = ['LINEAR']                                # Spatial correction filename
        #header['LINEAR'][:]  = [1.00, 0.00]                              # bruker_linearscale, bruker_linearoffset
        #header['PHD'][:]     = [0.68, 0.051]                             # Phosphor efficiency, phosphor thickness
        #header['OCTMASK'][:] = [0, 0, 0, 767, 767, 1791, 1023, 1023]
        
        # write the frame
        write_bruker_frame(self.path_sfrm, header, np.flipud(self.msk))
        
        # dump parameter dict
        self.parameter_dump()
    
    def imageHoverEvent(self, event):
        '''
        Show the position, pixel, and value under the mouse cursor.
        '''
        if event.isExit():
            self.statusBar.showMessage('')
            return
        pos = event.pos()
        x, y = pos.x(), pos.y()
        x = int(np.clip(x, 0, self.img.image.shape[1] - 1))
        y = int(np.clip(y, 0, self.img.image.shape[0] - 1))
        val = self.img.image[y, x]
        self.statusBar.showMessage(f"{val:>8} @ {x:>4} {y:>4}")
    
def bruker_header():
    '''
     default Bruker header
    '''
    import collections
    import numpy as np
    
    header = collections.OrderedDict()
    header['FORMAT']  = np.array([100], dtype=np.int64)                       # Frame Format -- 86=SAXI, 100=Bruker
    header['VERSION'] = np.array([18], dtype=np.int64)                        # Header version number
    header['HDRBLKS'] = np.array([15], dtype=np.int64)                        # Header size in 512-byte blocks
    header['TYPE']    = ['Some Frame']                                        # String indicating kind of data in the frame
    header['SITE']    = ['Some Site']                                         # Site name
    header['MODEL']   = ['?']                                                 # Diffractometer model
    header['USER']    = ['USER']                                              # Username
    header['SAMPLE']  = ['']                                                  # Sample ID
    header['SETNAME'] = ['']                                                  # Basic data set name
    header['RUN']     = np.array([1], dtype=np.int64)                         # Run number within the data set
    header['SAMPNUM'] = np.array([1], dtype=np.int64)                         # Specimen number within the data set
    header['TITLE']   = ['', '', '', '', '', '', '', '', '']                  # User comments (8 lines)
    header['NCOUNTS'] = np.array([-9999, 0], dtype=np.int64)                  # Total frame counts, Reference detector counts
    header['NOVERFL'] = np.array([-1, 0, 0], dtype=np.int64)                  # SAXI Format: Number of overflows
                                                                              # Bruker Format: #Underflows; #16-bit overfl; #32-bit overfl
    header['MINIMUM'] = np.array([-9999], dtype=np.int64)                     # Minimum pixel value
    header['MAXIMUM'] = np.array([-9999], dtype=np.int64)                     # Maximum pixel value
    header['NONTIME'] = np.array([-2], dtype=np.int64)                        # Number of on-time events
    header['NLATE']   = np.array([0], dtype=np.int64)                         # Number of late events for multiwire data
    header['FILENAM'] = ['unknown.sfrm']                                      # (Original) frame filename
    header['CREATED'] = ['01-Jan-2000 01:01:01']                              # Date and time of creation
    header['CUMULAT'] = np.array([20.0], dtype=np.float64)                    # Accumulated exposure time in real hours
    header['ELAPSDR'] = np.array([10.0, 10.0], dtype=np.float64)              # Requested time for this frame in seconds
    header['ELAPSDA'] = np.array([10.0, 10.0], dtype=np.float64)              # Actual time for this frame in seconds
    header['OSCILLA'] = np.array([0], dtype=np.int64)                         # Nonzero if acquired by oscillation
    header['NSTEPS']  = np.array([1], dtype=np.int64)                         # steps or oscillations in this frame
    header['RANGE']   =  np.array([1.0], dtype=np.float64)                    # Magnitude of scan range in decimal degrees
    header['START']   = np.array([0.0], dtype=np.float64)                     # Starting scan angle value, decimal deg
    header['INCREME'] = np.array([1.0], dtype=np.float64)                     # Signed scan angle increment between frames
    header['NUMBER']  = np.array([1], dtype=np.int64)                         # Number of this frame in series (zero-based)
    header['NFRAMES'] = np.array([1], dtype=np.int64)                         # Number of frames in the series
    header['ANGLES']  = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)      # Diffractometer setting angles, deg. (2Th, omg, phi, chi)
    header['NOVER64'] = np.array([0, 0, 0], dtype=np.int64)                   # Number of pixels > 64K
    header['NPIXELB'] = np.array([1, 2], dtype=np.int64)                      # Number of bytes/pixel; Number of bytes per underflow entry
    header['NROWS']   = np.array([512, 1], dtype=np.int64)                    # Number of rows in frame; number of mosaic tiles in Y; dZ/dY value
                                                                              # for each mosaic tile, X varying fastest
    header['NCOLS']   = np.array([512, 1], dtype=np.int64)                    # Number of pixels per row; number of mosaic tiles in X; dZ/dX
                                                                              # value for each mosaic tile, X varying fastest
    header['WORDORD'] = np.array([0], dtype=np.int64)                         # Order of bytes in word; always zero (0=LSB first)
    header['LONGORD'] = np.array([0], dtype=np.int64)                         # Order of words in a longword; always zero (0=LSW first
    header['TARGET']  = ['Mo']                                                # X-ray target material)
    header['SOURCEK'] = np.array([0.0], dtype=np.float64)                     # X-ray source kV
    header['SOURCEM'] = np.array([0.0], dtype=np.float64)                     # Source milliamps
    header['FILTER']  = ['?']                                                 # Text describing filter/monochromator setting
    header['CELL']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # Cell constants, 2 lines  (A,B,C,Alpha,Beta,Gamma)
    header['MATRIX']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # Orientation matrix, 3 lines
    header['LOWTEMP'] = np.array([1, -17300, -6000], dtype=np.int64)          # Low temp flag; experiment temperature*100; detector temp*100
    header['ZOOM']    = np.array([0.0, 0.0, 1.0], dtype=np.float64)           # Image zoom Xc, Yc, Mag
    header['CENTER']  = np.array([256.0, 256.0, 256.0, 256.0], dtype=np.float64) # X, Y of direct beam at 2-theta = 0
    header['DISTANC'] = np.array([5.0], dtype=np.float64)                     # Sample-detector distance, cm
    header['TRAILER'] = np.array([0], dtype=np.int64)                         # Byte pointer to trailer info (unused; obsolete)
    header['COMPRES'] = ['none']                                              # Text describing compression method if any
    header['LINEAR']  = np.array([1.0, 0.0], dtype=np.float64)                # Linear scale, offset for pixel values
    header['PHD']     = np.array([0.0, 0.0], dtype=np.float64)                # Discriminator settings
    header['PREAMP']  = np.array([1,1], dtype=np.int64)                       # Preamp gain setting
    header['CORRECT'] = ['UNKNOWN']                                           # Flood correction filename
    header['WARPFIL'] = ['Linear']                                            # Spatial correction filename
    header['WAVELEN'] = np.array([0.0, 0.0, 0.0], dtype=np.float64)           # Wavelengths (average, a1, a2)
    header['MAXXY']   = np.array([1, 1], dtype=np.int64)                      # X,Y pixel # of maximum counts
    header['AXIS']    = np.array([2], dtype=np.int64)                         # Scan axis (1=2-theta, 2=omega, 3=phi, 4=chi)
    header['ENDING']  = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float64)      # Setting angles read at end of scan
    header['DETPAR']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # Detector position corrections (Xc,Yc,Dist,Pitch,Roll,Yaw)
    header['LUT']     = ['lut']                                               # Recommended display lookup table
    header['DISPLIM'] = np.array([0.0, 0.0], dtype=np.float64)                # Recommended display contrast window settings
    header['PROGRAM'] = ['Python Image Conversion']                           # Name and version of program writing frame
    header['ROTATE']  = np.array([0], dtype=np.int64)                         # Nonzero if acquired by rotation (GADDS)
    header['BITMASK'] = ['$NULL']                                             # File name of active pixel mask (GADDS)
    header['OCTMASK'] = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)    # Octagon mask parameters (GADDS) #min x, min x+y, min y, max x-y, max x, max x+y, max y, max y-x
    header['ESDCELL'] = np.array([0.001, 0.001, 0.001, 0.02, 0.02, 0.02], dtype=np.float64) # Cell ESD's, 2 lines (A,B,C,Alpha,Beta,Gamma)
    header['DETTYPE'] = ['Unknown', 1.0, 1.0, 0, 0.1, 0.1, 1]                                           # Detector type
    header['NEXP']    = np.array([1, 0, 0, 0, 0], dtype=np.int64)             # Number exposures in this frame; CCD bias level*100,;
                                                                              # Baseline offset (usually 32); CCD orientation; Overscan Flag
    header['CCDPARM'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # CCD parameters for computing pixel ESDs; readnoise, e/ADU, e/photon, bias, full scale
    header['CHEM']    = ['?']                                                 # Chemical formula
    header['MORPH']   = ['?']                                                 # CIFTAB string for crystal morphology
    header['CCOLOR']  = ['?']                                                 # CIFTAB string for crystal color
    header['CSIZE']   = ['?']                                                 # String w/ 3 CIFTAB sizes, density, temp
    header['DNSMET']  = ['?']                                                 # CIFTAB string for density method
    header['DARK']    = ['NONE']                                              # Dark current frame name
    header['AUTORNG'] = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64) # Autorange gain, time, scale, offset, full scale
    header['ZEROADJ'] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)      # Adjustments to goniometer angle zeros (tth, omg, phi, chi)
    header['XTRANS']  = np.array([0.0, 0.0, 0.0], dtype=np.float64)           # Crystal XYZ translations
    header['HKL&XY']  = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) # HKL and pixel XY for reciprocal space (GADDS)
    header['AXES2']   = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)      # Diffractometer setting linear axes (4 ea) (GADDS)
    header['ENDING2'] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)      # Actual goniometer axes @ end of frame (GADDS)
    header['FILTER2'] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)      # Monochromator 2-theta, roll (both deg)
    header['LEPTOS']  = ['']
    header['CFR']     = ['']
    return header
    
def write_bruker_frame(fname, fheader, fdata):
    '''
     write a bruker image
    '''
    import numpy as np
    
    ########################
    ## write_bruker_frame ##
    ##     FUNCTIONS      ##
    ########################
    def pad_table(table, bpp):
        '''
         pads a table with zeros to a multiple of 16 bytes
        '''
        padded = np.zeros(int(np.ceil(table.size * abs(bpp) / 16)) * 16 // abs(bpp)).astype(_BPP_TO_DT[bpp])
        padded[:table.size] = table
        return padded
        
    def format_bruker_header(fheader):
        '''
         
        '''
        format_dict = {(1,   'int64'): '{:<71d} ',
                       (2,   'int64'): '{:<35d} {:<35d} ',
                       (3,   'int64'): '{:<23d} {:<23d} {:<23d} ',
                       (4,   'int64'): '{:<17d} {:<17d} {:<17d} {:<17d} ',
                       (5,   'int64'): '{:<13d} {:<13d} {:<13d} {:<13d} {:<13d}   ',
                       (6,   'int64'): '{:<11d} {:<11d} {:<11d} {:<11d} {:<11d} {:<11d} ',
                       (1,   'int32'): '{:<71d} ',
                       (2,   'int32'): '{:<35d} {:<35d} ',
                       (3,   'int32'): '{:<23d} {:<23d} {:<23d} ',
                       (4,   'int32'): '{:<17d} {:<17d} {:<17d} {:<17d} ',
                       (5,   'int32'): '{:<13d} {:<13d} {:<13d} {:<13d} {:<13d}   ',
                       (6,   'int32'): '{:<11d} {:<11d} {:<11d} {:<11d} {:<11d} {:<11d} ',
                       (1, 'float64'): '{:<71f} ',
                       (2, 'float64'): '{:<35f} {:<35f} ',
                       (3, 'float64'): '{:<23f} {:<23f} {:<23f} ',
                       (4, 'float64'): '{:<17f} {:<17f} {:<17f} {:<17f} ',
                       (5, 'float64'): '{:<13f} {:<13f} {:<13f} {:<13f} {:<15f} '}
    
        headers = []
        for name, entry in fheader.items():
            
            # TITLE has multiple lines
            if name == 'TITLE':
                name = '{:<7}:'.format(name)
                number = len(entry)
                for line in range(8):
                    if number < line:
                        headers.append(''.join((name, '{:<72}'.format(entry[line]))))
                    else:
                        headers.append(''.join((name, '{:<72}'.format(' '))))
                continue
    
            # DETTYPE Mixes Entry Types
            if name == 'DETTYPE':
                name = '{:<7}:'.format(name)
                string = '{:<20s} {:<11f} {:<11f} {:<1d} {:<11f} {:<10f} {:<1d} '.format(*entry)
                headers.append(''.join((name, string)))
                continue
            
            # format the name
            name = '{:<7}:'.format(name)
            
            # pad entries
            if type(entry) == list or type(entry) == str:
                headers.append(''.join(name + '{:<72}'.format(entry[0])))
                continue
            
            # fill empty fields
            if entry.shape[0] == 0:
                headers.append(name + '{:72}'.format(' '))
                continue
            
            # if line has too many entries e.g.
            # OCTMASK(8): np.int64
            # CELL(6), MATRIX(9), DETPAR(6), ESDCELL(6): np.float64
            # write the first 6 (np.int64) / 5 (np.float64) entries
            # and the remainder later
            if entry.shape[0] > 6 and entry.dtype == np.int64:
                while entry.shape[0] > 6:
                    format_string = format_dict[(6, str(entry.dtype))]
                    headers.append(''.join(name + format_string.format(*entry[:6])))
                    entry = entry[6:]
            elif entry.shape[0] > 5 and entry.dtype == np.float64:
                while entry.shape[0] > 5:
                    format_string = format_dict[(5, str(entry.dtype))]
                    headers.append(''.join(name + format_string.format(*entry[:5])))
                    entry = entry[5:]
            
            # format line
            format_string = format_dict[(entry.shape[0], str(entry.dtype))]
            headers.append(''.join(name + format_string.format(*entry)))
    
        # add header ending
        if headers[-1][:3] == 'CFR':
            headers = headers[:-1]
        padding = 512 - (len(headers) * 80 % 512)
        end = '\x1a\x04'
        if padding <= 80:
            start = 'CFR: HDR: IMG: '
            padding -= len(start) + 2
            dots = ''.join(['.'] * padding)
            headers.append(start + dots + end)
        else:
            while padding > 80:
                headers.append(end + ''.join(['.'] * 78))
                padding -= 80
            if padding != 0:
                headers.append(end + ''.join(['.'] * (padding - 2)))
        return ''.join(headers)
    ########################
    ## write_bruker_frame ##
    ##   FUNCTIONS END    ##
    ########################
    
    # assign bytes per pixel to numpy integers
    # int8   Byte (-128 to 127)
    # int16  Integer (-32768 to 32767)
    # int32  Integer (-2147483648 to 2147483647)
    # uint8  Unsigned integer (0 to 255)
    # uint16 Unsigned integer (0 to 65535)
    # uint32 Unsigned integer (0 to 4294967295)
    _BPP_TO_DT = {1: np.uint8,
                  2: np.uint16,
                  4: np.uint32,
                 -1: np.int8,
                 -2: np.int16,
                 -4: np.int32}
    
    # read the bytes per pixel
    # frame data (bpp), underflow table (bpp_u)
    bpp, bpp_u = fheader['NPIXELB']
    
    # generate underflow table
    # does not work as APEXII reads the data as uint8/16/32!
    if fheader['NOVERFL'][0] >= 0:
        data_underflow = fdata[fdata <= 0]
        fheader['NOVERFL'][0] = data_underflow.shape[0]
        table_underflow = pad_table(data_underflow, -1 * bpp_u)
        fdata[fdata < 0] = 0

    # generate 32 bit overflow table
    if bpp < 4:
        data_over_uint16 = fdata[fdata >= 65535]
        table_data_uint32 = pad_table(data_over_uint16, 4)
        fheader['NOVERFL'][2] = data_over_uint16.shape[0]
        fdata[fdata >= 65535] = 65535

    # generate 16 bit overflow table
    if bpp < 2:
        data_over_uint8 = fdata[fdata >= 255]
        table_data_uint16 = pad_table(data_over_uint8, 2)
        fheader['NOVERFL'][1] = data_over_uint8.shape[0]
        fdata[fdata >= 255] = 255

    # shrink data to desired bpp
    fdata = fdata.astype(_BPP_TO_DT[bpp])
    
    # write frame
    with open(fname, 'wb') as brukerFrame:
        brukerFrame.write(format_bruker_header(fheader).encode('ASCII'))
        brukerFrame.write(fdata.tobytes())
        if fheader['NOVERFL'][0] >= 0:
            brukerFrame.write(table_underflow.tobytes())
        if bpp < 2 and fheader['NOVERFL'][1] > 0:
            brukerFrame.write(table_data_uint16.tobytes())
        if bpp < 4 and fheader['NOVERFL'][2] > 0:
            brukerFrame.write(table_data_uint32.tobytes())
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()
