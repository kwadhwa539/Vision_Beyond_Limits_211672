#!/bin/bash 

set -euo pipefail

# this function is called when Ctrl-C is sent
function trap_ctrlc ()
{
    # perform cleanup here
    echo "Ctrl-C or Error caught...performing clean up"

    if [ -d "$input"/spacenet_gt ]; then
           rm -rf "$input"/spacenet_gt
    fi

    exit 99
}

# initialise trap to call trap_ctrlc function
# when signal 2 (SIGINT) is received
trap "trap_ctrlc" 2 9 13 3

help_message () {
        printf "${0}: Moves files around for the spacenet model to train\n\t-i /path/to/xBD/ \n\t-s split percentage to go to train\n\t-x /path/to/xview-2/repository/\n\t(Note: this script expects mask_polygons.py to have ran first to create labels)\n\n"
}

# Checking for `bc` first (users reported that wasn't installed on some systems)
if ! [ -x "$(command -v bc)" ]; then
  echo 'Error: bc is not installed, please install before continuing.' >&2
  exit 98
fi

if [ $# -lt 3 ]; then 
        help_message
        exit 1
fi

while getopts "i:s:x:h" OPTION
do
     case $OPTION in
         h)
             help_message
             exit 1
             ;;
         i)
             input="$OPTARG"
             ;;
         s)
             split="$OPTARG"
             ;;
         x)
             XBDIR="$OPTARG"
             ;;
     esac
done


# Get list of disasters to iterate over 
disasters=`/bin/ls -1 "$input"`

# Making the spacenet training directory 
mkdir -p "$input"/spacenet_gt/images
mkdir -p "$input"/spacenet_gt/labels
mkdir -p "$input"/spacenet_gt/dataSet

# for each disaster, copy the pre images and labels to the spacenet training directory
for disaster in $disasters; do
    masks=`/bin/ls -1 "$input"/"$disaster"/masks`
    for mask in $masks; do
        cp "$input"/"$disaster"/masks/$mask "$input"/spacenet_gt/labels
        cp "$input"/"$disaster"/images/$mask "$input"/spacenet_gt/images
    done
done

# Listing all files to do the split
cd "$input"/spacenet_gt/dataSet/
touch all_images.txt
/bin/ls -1 "$input"/spacenet_gt/images > all_images.txt

line_count=`cat all_images.txt | wc -l`
lines_to_split=$(bc -l <<< "$line_count"*"$split")
split -l `awk -F. '{print $1}' <<< $lines_to_split` all_images.txt

mv ./xaa train.txt
mv ./xab val.txt
rm all_images.txt

# Running the mean creation code over the images
python "$XBDIR"/spacenet/src/features/compute_mean.py "$input"/spacenet_gt/dataSet/train.txt --root "$input"/spacenet_gt/images/ --output "$input"/spacenet_gt/dataSet/mean.npy 

echo "Done!" 
