def frontiers_from_time_to_bar(seq, bars):
    """
    Converts the frontiers in time to a bar index.
    The selected bar is the one which end is the closest from the frontier.

    Parameters
    ----------
    seq : list of float
        The list of frontiers, in time.
    bars : list of tuple of floats
        The bars, as (start time, end time) tuples.

    Returns
    -------
    seq_barwise : list of integers
        List of times converted in bar indexes.

    """
    seq_barwise = []
    for frontier in seq:
        for idx, bar in enumerate(bars):
            if frontier >= bar[0] and frontier < bar[1]:
                if bar[1] - frontier < frontier - bar[0]:
                    seq_barwise.append(idx)
                else:
                    if idx == 0:
                        seq_barwise.append(idx)
                        #print("The current frontier {} is labelled in the start silence ({},{}), which is incorrect.".format(frontier, bar[0], bar[1]))
                    else:
                        seq_barwise.append(idx - 1)
                break
    return seq_barwise

def get_segmentation_from_txt(path, annotations_type):
    """
    Reads the segmentation annotations, and returns it in a list of tuples (start, end, index as a number)
    This function has been developped for AIST and MIREX10 annotations, adapted for these types of annotations.
    It will not work with another set of annotation.

    Parameters
    ----------
    path : String
        The path to the annotation.
    annotations_type : "AIST" [1] or "MIREX10" [2]
        The type of annotations to load (both have a specific behavior and formatting)
        
    Raises
    ------
    err.InvalidArgumentValueException
        If the type of annotations is neither AIST or MIREX10

    Returns
    -------
    segments : list of tuples (float, float, integer)
        The segmentation, formatted in a list of tuples, and with labels as numbers (easier to interpret computationnally).

    References
    ----------
    [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
    
    [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
    Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

    """
    file_seg = open(path)
    segments = []
    labels = []
    for part in file_seg.readlines():
        tupl = part.split("\t")
        if tupl[2] not in labels: # If label wasn't already found in this annotation
            idx = len(labels)
            labels.append(tupl[2])
        else: # If this label was found for another segment
            idx = labels.index(tupl[2])
        if annotations_type == "AIST":
            segments.append(((int(tupl[0]) / 100), (int(tupl[1]) / 100), idx))
        elif annotations_type == "MIREX10":
            segments.append((round(float(tupl[0]), 3), round(float(tupl[1]), 3), idx))
        else:
            print("Annotations type not understood")
    return segments
