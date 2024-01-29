#!/bin/sh                                                                       
#                                                                               
#./run_pattern_sim.sh $NAME $GEOM_FILE $PDB_FILE $INTENSITY_FILE                
# $TASKS $PARTITION $QOS $HOURS $TAG                                            
#                                                                               
NAME=$1                                                                         
GEOM_FILE=$2                                                                    
PDB_FILE=$3                                                                     
INTENSITY_FILE=$4                                                               
TASKS=$5                                                                        
PARTITION=$6                                                                    
QOS=$7                                                                          
HOURS=$8                                                                        
TAG=$9                                                                          
                                                                                
INPUT=${NAME}.hkl    #??                                                        
CRYSTAL=1IC6.cell     #??                                                                    
NUM_PATTERNS=10000   #10,000                                                    
POINTGROUP=4/mmm                                                                
CRYSTAL_MIN=10000                                                               
CRYSTAL_MAX=10000                                                               
SPECTRUM=tophat                                                                 
SAMPLING=7                                                                      
BANDWIDTH=0.01                                                                  
N_PHOTONS=3e8                                                                   
BEAM_RADIUS=5e-6                                                                
                                                                                
RUN="$NAME_$TAG"                                                                  
STREAM="$RUN.stream"                                                            
SLURMFILE="$RUN.sh"                                                             
                                                                                
    echo "#!/bin/sh" > $SLURMFILE                                               
    echo >> $SLURMFILE                                                          
                                                                                
    echo "#SBATCH --time=0-60:00" >> $SLURMFILE                                 
    echo "#SBATCH --ntasks=$TASKS" >> $SLURMFILE                                
    echo >> $SLURMFILE                                                          
                                                                                
    echo "#SBATCH --chdir   $PWD" >> $SLURMFILE                                 
    echo "#SBATCH --job-name  $RUN" >> $SLURMFILE                               
    echo "#SBATCH --output    $RUN.out" >> $SLURMFILE                           
    echo "#SBATCH --error    $RUN.err" >> $SLURMFILE                            
    echo >> $SLURMFILE                                                          
                                                                                
    #CHANGE AS NEEDED                                                           
    command="pattern_sim"                                                       
    command="$command -g $GEOM_FILE"             #geom                          
    command="$command -p $PDB_FILE"              #pdb                           
    command="$command --number=$NUM_PATTERNS"    #number of patterns            
    command="$command -o $RUN"                   #output                        
    command="$command -i $INTENSITY_FILE"        #.hkl input                    
    command="$command -y $POINTGROUP"            #pointgroup                    
    command="$command -r"                                                       
    command="$command --min-size=$CRYSTAL_MIN"                                  
    command="$command --max-size=$CRYSTAL_MAX" #crystal min/max size            
    command="$command --spectrum=$SPECTRUM"      # spectrum                     
    command="$command -s $SAMPLING"              # sampling                     
    command="$command --background=0"                                           
    command="$command --beam-bandwidth=$BANDWIDTH" #beam bandwidth              
    command="$command --nphotons=$N_PHOTONS"           #n_photons               
    command="$command --beam-radius=$BEAM_RADIUS"       #beam radius            
                                                                                
    #EXTRA OPTIONS                                                              
   # command="$command --no-noise"                                              
   # command="--really-random"                                                  
   # command="--no-fringes"                                                     
   # command="--no-images" #comment out -o then                                 
                                                                                
    echo $command >> $SLURMFILE                                                 
                                                                                
    sbatch -p $PARTITION -q $QOS -t $HOURS:00:00 $SLURMFILE                     
#    sbatch -p fn1 $SLURMFILE                                                   
#    sbatch -p phi $SLURMFILE 