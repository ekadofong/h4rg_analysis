

def jhu2tiger ( pathtofile ):
    jhu_root = "/data/"
    tiger_root = "/projects/HSC/PFS/JHU/"
    return tiger_root + pathtofile.strip(jhu_root).replace('83.fits', '23.fits')