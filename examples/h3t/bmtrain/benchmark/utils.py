def format_size_(x):
    if x < 1024:
        return "{:d}B".format(x)
    if x < 1024 * 1024:
        return "{:4.2f}KB".format(x / 1024)
    if x < 1024 * 1024 * 1024:
        return "{:4.2f}MB".format(x / 1024 / 1024)
    return "{:4.2f}GB".format(x / 1024 / 1024 / 1024)

def format_size(x):
    return "{:.6s}".format(format_size_(x))