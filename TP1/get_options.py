import sys
import getopt

def get_options_from_cmd(argv):
    """Get the options input from user at execution in cmd"""
    options = {"return_statistics": False}
    arg_help = "{0}\n -h: help \n -s: Print the statistics for the compression algorithm ".format(argv[0])
    try:
        opts, args = getopt.getopt(argv[1:], "hs", ["help", "print_statistics"])
    except:
        print(arg_help)
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-s", "--print_statistics"):
            options["return_statistics"] = True     
    return options