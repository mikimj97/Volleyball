import os
import logging
import re
import datetime

logger = logging.getLogger(__name__)


def gather_ml_results(args):
    """
    Gathers primary results from the disperate folders after running and evaluating a model.
    Args:
        args: a Namespace of variables to be used for setting the parameters for preprocessing
    Returns:
        Nothing.   But writes a file containing results in the top level results directory.
    """
    results_dir = args.completed_results_dir
    logger.info("gathering results from {}".format(results_dir))
    # open and create a results file.
    datenow = datetime.date.today()
    outputfilename = "results_summary.txt".format(results_dir)
    logger.info("writing results summary to {}".format(outputfilename))
    outputfile = open(os.path.join(results_dir, outputfilename), 'a')
    # go to the results directory and recursively traverse subdirectories until we get to a leaf.
    # a leaf is a directory with no subdirectories.
    # see https://stackoverflow.com/questions/16953842/using-os-walk-to-recursively-traverse-directories-in-python
    for root, dirs, files in os.walk(results_dir):
        path = root.split(os.sep)
        # print(path[-1][:4])

        if "old_results" in root or "testing" in root or "no_" in root:
            continue

        if not (path[-1][:4] == "full" or path[-1][:4] == "jump"):
            continue

        # print ((len(path) - 1) * '---', os.path.basename(root))
        outputfile.write(root + "\n")

        for file in files:
            # print(len(path) * '---', file)
            # we have a file.  is it a results file?
            # print(os.path.join(root, file))

            if (is_a_results_file(file)) and not (file == outputfilename):
                # it is.  Process it.
                logger.info("found results file in {}".format(os.path.join(*path, file)))
                process_results(path, file, outputfile)
        # end of line for this particular result.
        # outputfile.write("\n------------------------\n")
    outputfile.close()


def is_a_results_file(filename):
    """
    determines if this filename is likely a file containing results.
    text files ending .txt are results files.
    Args:
         filename: a string.
    Returns:
         True if the filename contains a bunch of digits at the endends in .txt
    """
    matcher = re.compile(".txt")
    # print (filename)
    matches = matcher.findall(filename)
    return (len(matches) > 0)


def process_results(path_pieces, resultsfile_name, outputfile):
    """
    grabs the results of the file resultsfile_name at the path created
    by assembling path_pieces and writes it to the given outputfile
    no return value.
    """
    # get all of our information organized.
    full_path_to_results = os.path.join(*path_pieces, resultsfile_name)
    # open the file
    thefile = open(full_path_to_results, 'r')
    filecontents = (thefile.read())
    cleancontents = filecontents.replace('"', "")
    thefile.close()
    outputfile.write(cleancontents + ", " + resultsfile_name + "\n")
    return
