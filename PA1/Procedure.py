from DataFrame import DataFrame

class Procedure():
    def __init__(self, name):
        
        self.name = name
        self.dataframes = []

        # Find data file paths
        calbody = f"{self.name}-CALBODY.TXT"
        calreadings = f"{self.name}-CALREADINGS.TXT"
        empivot = f"{self.name}-EMPIVOT.TXT"
        optpivot = f"{self.name}-OPTPIVOT.TXT"

        # Load calibration data from the optical tracker
        with open(calbody) as f:
            linesCalBody = f.readlines()
        metadata = linesCalBody[0].split(",")
        self.Nd = int(metadata[0])
        self.Na = int(metadata[1])
        self.Nc = int(metadata[2])

        # Load calibration data from the EM tracker
        with open(calreadings) as f:
            linesCalReadings = f.readlines()
        self.Nf = int(linesCalReadings[0].split(",")[3])

        # Load data about the EM post from the EM tracker
        with open(empivot) as f:
            linesEmPivot = f.readlines()
        metadata = linesEmPivot[0].split(",")
        self.Ng = int(metadata[0])

        # Load data about the optical post from the optical tracker
        with open(optpivot) as f:
            linesOpPivot = f.readlines()
        metadata = linesOpPivot[0].split(",")
        Nh = int(metadata[1])

        # Calculate the size of each data slice
        scal = (self.Nd + self.Na + self.Nc)
        sem = (self.Ng)
        sop = (self.Nd + Nh)

        # Create a DataFrame for each frame number
        for i in range(self.Nf):
            df = DataFrame(i)
            df.calibrate(self.Nd, self.Na, self.Nc, linesCalBody[1:])
            df.calSensor(self.Nd, self.Na, self.Nc, linesCalReadings[i*scal+1:(i+1)*scal+1])
            df.emProbe(self.Ng, linesEmPivot[i*sem+1:(i+1)*sem+1])
            df.opProbe(self.Nd, Nh, linesOpPivot[i*sop+1:(i+1)*sop+1])
            self.dataframes.append(df)
        

    def compute_expected(self):
        """Compute the expected positions of the calibration object's EM markers for each frame"""
        
        Ci = []
        for frame in self.dataframes:
            Ci.append(frame.compute_expected())
        return Ci

    def pivot_calibration(self, probeType):
        """Perform a pivot calibration"""

        if probeType.upper() == "EM":
            probe0 = self.dataframes[0].emProbeCloud
            probes = [df.emProbeCloud for df in self.dataframes[1:]]
        elif probeType.upper() == "OP":
            probe0 = self.dataframes[0].opProbeCloud
            probes = [df.opProbeCloud for df in self.dataframes[1:]]
        else:
            raise Exception("ProbeType must be either EM or OP for the \
                electromagnetic probe or optical probe, respectively.")
        
        return probe0.pivot(probes)
    

    def compile_output(self):
        """Create an output file for Programming Assignment 1"""
        
        # Perform pivot calibrations and compute expected Ci
        [pt_em, p_piv_em] = self.pivot_calibration("em")
        [pt_op, p_piv_op] = self.pivot_calibration("op")
        Ci = self.compute_expected()

        # Create an output file (or overwrite the existing one)
        outputpath = self.name.split("/")
        outputpath[-2] = "output"
        outputpath = outputpath.join()
        filename = inputpath[-1]
        
        f = open(f"{outputpath}-OUTPUT-1.txt", "w")
        
        f.write(f"{self.Nc}, {self.Nf}, {filename}-OUTPUT-1.txt\n")
        f.close()

        # Append the results to the output file
        f = open(f"{outputpath}-a-OUTPUT-1.txt", "a")
        f.write(f"  %.2f,   %.2f,   %.2f\n" % (pt_em[0,0], pt_em[1,0], pt_em[2,0]))
        f.write(f"  %.2f,   %.2f,   %.2f\n" % (pt_op[0,0], pt_op[1,0], pt_op[2,0]))
        for df in Ci:
            for c in df:
                f.write(f"  %.2f,   %.2f,   %.2f\n" % (c[0], c[1], c[2]))
        f.close()


    def __repr__(self):
        if name:
            return f"Procedure(name={self.name})"