import sys, os
import calc_Bj as p4
import distortion_correction as d
import calc_Freg as p5
import compute_tip_loc as p6
import test


def main():
    """
    Main method, takes command line arguments for input files.
    :return: None
    """

    # Initialize variables
    calbody = None
    calreadings = None
    empivot = None
    ctfiducials = None
    emfiducialss = None
    emnav = None

    # Add 'test' command line option
    if str(sys.argv[1]) == 'test':
        if len(sys.argv) == 3:
            empivot = sys.argv[2]
        else:
            empivot = "PA12 - Student Data/pa2-debug-a-empivot.txt"
        if len(sys.argv) == 4:
            tolerance = float(sys.argv[3])
            test.test_reg(tolerance)
            test.test_pivot_cal(empivot, tolerance)
            test.test_normalization()
            test.test_f()
            test.test_solve_linear_sys(tolerance)

        else:
            test.test_reg()
            test.test_pivot_cal(empivot)
            test.test_normalization()
            test.test_f()
            test.test_solve_linear_sys()

        sys.exit(0)

    # Parse arguments for regular execution
    for arg in sys.argv:
        if arg.split('.')[0].split('-')[-1] == 'calbody':
            calbody = arg
        if arg.split('.')[0].split('-')[-1] == 'calreadings':
            calreadings = arg
        if arg.split('.')[0].split('-')[-1] == 'empivot':
            empivot = arg
        if arg.split('.')[0].split('-')[-1] == 'fiducials':
            ctfiducials = arg
        if arg.split('.')[0].split('-')[-1] == 'fiducialss':
            emfiducialss = arg
        if arg.split('.')[0].split('-')[-1] == 'nav':
            emnav = arg

    outname = sys.argv[1].split('/')[-1].rsplit('-', 1)[0] + '-output2.txt'

    # Run code for probelms 4 - 6 and save output
    tofile(outname, calbody, calreadings, empivot, ctfiducials, emfiducialss, emnav)


def tofile(outfile, calbody, calreadings, empivot, ctfiducials, emfiducialss, emnav):
    """
    Runs methods for questions 4-6 and writes output file with solutions.
    :param outfile: File name/path for output file
    :param calbody: File name/path for the calibration object data file
    :param calreadings: File name/path for the readings from the trackers
    :param empivot: File name/path for EM pivot poses
    :param ctfiducials: The file name/path of the file with positions of each fiducial pin in the CT frame
    :param emfiducialss: The file name/path of the file with marker positions when the pointer is on the fiducials,
                         relative to the EM tracker.
    :param emnav: The file name/path of the file with marker positions when the pointer is in an arbitrary position,
                  relative to the EM tracker.

    :type outfile: str
    :type calbody: str
    :type calreadings: str
    :type empivot: str
    :type ctfiducials: str
    :type emfiducialss: str
    :type emnav: str

    :return: None
    """
    p_ans, C, qmi, qma, qmis, qmas = d.distortion_calculation(calbody, calreadings, empivot)

    Cs = p4.tip_in_EM(empivot, emfiducialss, p_ans[0], C, qmi, qma, qmis, qmas)

    F = p5.find_freg(ctfiducials, Cs)

    CT = p6.tip_pointer(empivot, emnav, p_ans[0], F, C, qmi, qma, qmis, qmas)

    f = open(outfile, 'w')
    h, t = os.path.split(outfile)
    f.write('{0}, {1}\n'.format(CT.data.shape[1], t))
    for i in range(CT.data.shape[1]):
        f.write('{0:>10},{1:>10},{2:>10}\n'.format(format(CT.data[0][i], '.2f'),
                                               format(CT.data[1][i], '.2f'),
                                               format(CT.data[2][i], '.2f')))
    f.close()


if __name__ == '__main__':
    main()
