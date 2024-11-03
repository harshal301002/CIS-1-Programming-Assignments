from Marker import Marker
from FrameOperations import Frame
from PointCloud import PointCloud
import numpy as np

class DataFrame():
    """
    TODO: Description here
    """

    def __init__(self, fnum):
        self.fnum = fnum              # Frame number in procedure
        self.Nd = 0                           # Number of optical markers on the EM Tracker
        self.Na = 0                           # Number of optical markers on the calibration object
        self.Nc = 0                           # Number of EM markers on calibration object
        self.Ng = 0                           # Number of EM markers on the EM post
        self.Nh = 0                           # Number of optical markers on optical post
        self.emObjects = {}                   # Collection of markers registered in the EM tracking system
        self.opObjects = {}                   # Collection of markers registered in the optical tracking system
        self.calObjects = {}                  # Collection of markers registered in the calibration object system


    def __repr__(self):
        return f"DataFrame(fnum={self.fnum})"


    def getPosition(self, line):
        """Retrieve nav element coordinates from a data file line"""
        pos = line.split(",")
        return (float(pos[0]), float(pos[1]), float(pos[2]))


    def get_points(self, name, system):
        """Retrieve an array of position vectors for a given set of markers with respect to a given system"""
        return np.array([m.position for key,m in system.items() if name in key]).T


    @property
    def Fd(self):
        # Create point clouds of the emTracker LED markers from both EM and optical sensor data
        emTracker_em = self.get_points(name="emTrackerOpMarker", system=self.emObjects)
        emTracker_op = self.get_points(name="emTrackerOpMarker", system=self.opObjects)
        emCloud = PointCloud(emTracker_em)
        opCloud = PointCloud(emTracker_op)
        
        # Compute transformation from optical sensor coordinates to EM coordinates
        Fd = emCloud.register(opCloud)
        return Fd

    
    @property
    def emProbeCloud(self):
        """PointCloud of the EM post marker positions in EM system coordinates"""
        return PointCloud(self.get_points(name="emPostMarker", system=self.emObjects))


    @property
    def opProbeCloud(self):
        """PointCloud of the optical post marker positions in EM system coordinates"""
        opPoints = self.get_points(name="opPostMarker", system=self.opObjects)
        return PointCloud(self.Fd.apply_to_cloud(opPoints))


    def calibrate(self, Nd, Na, Nc, data):
        """Use calibration data to add markers to the EM tracker and calibration object"""
        self.Nd = Nd
        self.Na = Na
        self.Nc = Nc

        # Initialize markers using the position data
        for d in range(self.Nd):
            self.emObjects.update({f"emTrackerOpMarker{d}": Marker(name=f"emTrackerOpMarker{d}", \
                position=self.getPosition(data[d]))})
        for a in range(self.Na):
            self.calObjects.update({f"calOpMarker{a}": Marker(name=f"calOpMarker{a}", \
                position=self.getPosition(data[a + Nd]))})
        for c in range(self.Nc):
            self.calObjects.update({f"calEmMarker{c}": Marker(name=f"calEmMarker{c}", \
                position=self.getPosition(data[c + Na + Nd]))})
    

    def calSensor(self, Nd, Na, Nc, data):
        """Use calibration data to add markers to the EM tracker and calibration object"""
        self.Nd = Nd
        self.Na = Na
        self.Nc = Nc

        # Initialize markers using the position data
        for d in range(self.Nd):
            self.opObjects.update({f"emTrackerOpMarker{d}": Marker(name=f"emTrackerOpMarker{d}", \
                position=self.getPosition(data[d]))})
        for a in range(self.Na):
            self.opObjects.update({f"calOpMarker{a}": Marker(name=f"calOpMarker{a}", \
                position=self.getPosition(data[a + Nd]))})
        for c in range(self.Nc):
            self.emObjects.update({f"calEmMarker{c}": Marker(name=f"calEmMarker{c}", \
                position=self.getPosition(data[c + Na + Nd]))})

    
    def emProbe(self, Ng, data):
        """Use EM tracker data to add EM markers the EM post"""
        self.Ng = Ng
        
        # Initialize markers using the position data
        for g in range(self.Ng):
            self.emObjects.update({f"emPostMarker{g}": (Marker(name=f"emPostMarker{g}", \
                position=self.getPosition(data[g])))})


    def opProbe(self, Nd, Nh, data):
        """Use optical tracker data to add LED markers to the EM tracker and optical post"""
        self.Nd = Nd
        self.Nh = Nh

        # Initialize markers using the position data
        for d in range(self.Nd):
            self.opObjects.update({f"emTrackerOpMarker{d}": Marker(name=f"emTrackerOpMarker{d}", \
                position=self.getPosition(data[d]))})
            
        for h in range(self.Nh):
            self.opObjects.update({f"opPostMarker{h}": Marker(name=f"opPostMarker{h}", \
                position=self.getPosition(data[h + Nd]))})
    

    def compute_expected(self):
        """Compute the expected positions of the calibration object's EM markers"""

        # Create point clouds of the calibration object LED markers from both optical and calibration data
        calObj_cal = self.get_points(name="calOpMarker", system=self.calObjects)
        calObj_op = self.get_points(name="calOpMarker", system=self.opObjects)
        calCloud = PointCloud(calObj_cal)
        opCloud = PointCloud(calObj_op)
        
        # Compute transformation from calibration object coordinates to optical coordinates
        Fa = calCloud.register(opCloud)

        # Compute the calibration object EM markers expected positions with respect to the EM tracker
        ci = self.get_points(name="calEmMarker", system=self.calObjects)
        Ci = self.Fd.inv.compose(Fa).apply_to_cloud(ci)

        # Return expected Ci data
        Ci_actual = self.get_points(name="calEmMarker", system=self.emObjects)
        return Ci.T