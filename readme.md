# Optical Music Sheet Image Recognition  
The application uses various Python libraries to read either a PDF document or image file, recognize the staff lines, bar lines, clefs, and then reads the music sheet in its entirety. It then maps the array of notes to standard MIDI keys and outputs the corresponding audio for piano and violin instruments.  
More about how to use the app is documented in `report.pdf`.
## Reading Input  
The app accepts both PDF documents and images. The underlying structure is capable of identifying different musical symbols in images, so if a PDF document is given as an input, the `pdf2image` library is used to convert the file into an image with appropriate dimensions.  
## Processing the Image  
At first, staff lines are identified using black pixel density in each pixel row. The ones with more density than a threshold are considered to be staff lines. The distance between staff lines is measured by vertical traversing a pixel column between two recognized staff lines. Staff lines are grouped in five and that forms the basis for the next step of recognition.  
After correctly identifying the staff lines, the bar lines are then found using the same method only with the difference that here we are measuring vertical pixel density. Note that in this case we only measure the density between the upper staff line and lower staff line.  
After finding the structure for which notes are written on, clefs, notes and other symbols are found using different experimental thresholds compared to the corresponding images in `asset` folder. All the symbols are then sorted using their x-coordinates. The y-coordinates for different notes is used to identify their corresponding note.  
The connected eight-notes are recognized by comparing the black pixel density in the middle of the line connecting their flags. If black pixel density surpasses the given threshold, the notes are identified as connected eight-notes and otherwise as quarter notes (Note that other than the connecting flag the symbols are identical).  
Finally, after sorting the notes, identifying sharps and flats and assigning them to their nearest note, we now map the notes to their MIDI key and form the resulting array. Note that the clef, and the tempo are taken care of in the MIDI file by setting their corresponding parameters.  
## Output  
The MIDI file is produced using `MIDIFile` library, two instances of audio with different instruments are created, the first is piano and the second the is violin. The files are created with `music21` library and then save in the `output` directory. All other images from every step of the process e.g. staff lines recognition are found in the `output` folder. The files are overwritten once a new music sheet is given to the program.

Note that some methods and techniques for implementation are inspired from (my thanks to both of them):
- [Mozart by aashrafh](https://github.com/aashrafh/Mozart)
- [cadenCV by afikanyati](https://github.com/afikanyati/cadenCV)  

Dependencies for the project are given in the `report.pdf`.
