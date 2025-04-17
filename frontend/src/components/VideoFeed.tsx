import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Camera, Play, Square, Maximize2, Minimize2, Shield } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { cn } from "@/lib/utils";
import { config } from '../config';

// Types (Keep existing or adjust as needed)
interface BBox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

interface Detection {
  id: number;
  type: string;
  confidence: number;
  severity: 'low' | 'medium' | 'high';
  timestamp: Date;
  bbox: BBox; // Keep or adjust if needed for onDetection prop
  associatedPersonId?: string;
}

interface YoloPerson {
  id: number;
  bbox: number[]; // [xmin, ymin, xmax, ymax]
}

interface MmdetGun {
  bbox: BBox;
  class: string;
  confidence: number;
}

interface PersonGunState {
  [personId: string]: { 
    has_gun: boolean; 
    first_detected_frame: number 
  };
}

interface NewAlert {
  personId: string;
  gunConfidence: number;
}

// Component
const VideoFeed = ({ onDetection }: { onDetection: (detection: Detection) => void }) => {
  const { toast } = useToast();
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isScanning, setIsScanning] = useState(true); // Default to scanning

  // --- State for HTML Overlay Rendering ---
  const [yoloPersons, setYoloPersons] = useState<YoloPerson[]>([]);
  const [mmdetGuns, setMmdetGuns] = useState<MmdetGun[]>([]);
  const [personGunState, setPersonGunState] = useState<PersonGunState>({});
  const [currentFrameSrc, setCurrentFrameSrc] = useState<string | null>(null);
  const [naturalDimensions, setNaturalDimensions] = useState<{ width: number; height: number } | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  // --- --- 

  useEffect(() => {
    let wsInstance: WebSocket | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectInterval = 3000; // 3 seconds

    const connectWebSocket = () => {
      console.log('Attempting to connect to WebSocket:', config.videoFeedWsUrl);
      wsInstance = new WebSocket(config.videoFeedWsUrl);
      setWs(wsInstance); // Store instance

      wsInstance.onopen = () => {
        console.log('Video WebSocket connection established');
        setIsPlaying(true);
        setError(null);
        reconnectAttempts = 0;
      };

      wsInstance.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'video_frame' && message.payload) {
            const {
              image,
              yolo_persons,
              mmdet_guns,
              person_gun_state,
              new_high_confidence_alerts
            } = message.payload;

            // Update state for rendering overlays
            setYoloPersons(yolo_persons || []);
            setMmdetGuns(mmdet_guns || []);
            setPersonGunState(person_gun_state || {});
            setCurrentFrameSrc(`data:image/jpeg;base64,${image}`); // Update image source

            // --- Trigger Notifications based *only* on new_high_confidence_alerts ---
            if (new_high_confidence_alerts && Array.isArray(new_high_confidence_alerts) && new_high_confidence_alerts.length > 0) {
              new_high_confidence_alerts.forEach((alert: NewAlert) => {
                toast({
                  title: `ALERT: High-Confidence Gun Detected!`,
                  description: `Person: ${alert.personId}, Confidence: ${(alert.gunConfidence * 100).toFixed(0)}%`,
                  variant: "destructive"
                });

                // Optional: Call onDetection prop
                const processedDetection: Detection = {
                  id: Date.now() + Math.random(),
                  type: 'gun',
                  confidence: alert.gunConfidence,
                  severity: 'high',
                  timestamp: new Date(),
                  bbox: { xmin: 0, ymin: 0, xmax: 0, ymax: 0 }, // Default bbox
                  associatedPersonId: alert.personId
                };
                onDetection(processedDetection);
              });
            }
          }
        } catch (err) {
          console.error('Error processing WebSocket message:', err);
          setError('Error processing video data');
        }
      };

      wsInstance.onerror = (err) => {
        console.error('Video WebSocket error:', err);
        setError('WebSocket connection error');
      };

      wsInstance.onclose = (event) => {
        console.log('Video WebSocket connection closed:', event);
        setCurrentFrameSrc(null); // Clear image on disconnect
        setNaturalDimensions(null);
        setYoloPersons([]);
        setMmdetGuns([]);
        setPersonGunState({});
        setIsPlaying(false); // Also update playing state

        if (reconnectAttempts < maxReconnectAttempts) {
          setTimeout(() => {
            console.log(`Reconnecting... Attempt ${reconnectAttempts + 1}`);
            reconnectAttempts++;
            connectWebSocket();
          }, reconnectInterval);
        } else {
          console.error('Max reconnect attempts reached.');
          setError('WebSocket disconnected. Max reconnect attempts reached.')
        }
      };
    };

    connectWebSocket();

    return () => {
      if (wsInstance) {
        console.log('Cleaning up WebSocket connection');
        wsInstance.close();
        setWs(null);
      }
    };
  }, []); // Run only once on mount

  // Effect to handle setting natural dimensions when image source changes
  useEffect(() => {
    const img = imageRef.current;
    if (img && currentFrameSrc) {
      const handleLoad = () => {
        // Set dimensions only if they haven't been set or if they change
        // This simple check assumes video dimensions are constant.
        if (!naturalDimensions || naturalDimensions.width !== img.naturalWidth || naturalDimensions.height !== img.naturalHeight) {
           console.log(`Setting natural dimensions: ${img.naturalWidth}x${img.naturalHeight}`);
           setNaturalDimensions({ width: img.naturalWidth, height: img.naturalHeight });
        }
      };

      // Add event listener
      img.addEventListener('load', handleLoad);
      // Set src AFTER attaching listener to ensure load event fires correctly
      img.src = currentFrameSrc;

      // Cleanup function
      return () => {
        img.removeEventListener('load', handleLoad);
      };
    }
  }, [currentFrameSrc, naturalDimensions]); // Update dependency array slightly

  // --- Helper Functions for UI ---
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    // Basic fullscreen toggle (might need more robust implementation)
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  };

  const togglePlay = () => {
    // Note: Actual play/pause might need interaction with WebSocket start/stop if implemented
    setIsPlaying(!isPlaying);
    toast({
      title: isPlaying ? "Video paused (display only)" : "Video playing (display only)",
      variant: "default"
    });
  };

  const toggleScanning = () => {
    // Note: Actual scanning toggle needs communication back to backend via WebSocket or API
    setIsScanning(!isScanning);
    toast({
      title: isScanning ? "Threat detection paused (display only)" : "Threat detection activated (display only)",
      variant: "default"
    });
  };
  // --- --- 

  return (
    <div className={cn(
      "flex flex-col bg-card border border-border rounded-lg overflow-hidden",
      isFullscreen ? "fixed inset-0 z-50" : "h-full"
    )}>
      {/* Header */} 
      <div className="flex items-center justify-between p-3 border-b border-border">
        <div className="flex items-center space-x-2">
          <Camera className="h-5 w-5 text-muted-foreground" />
          <h2 className="font-semibold text-sm">Live Feed</h2>
        </div>
        <div className="flex items-center space-x-1">
          {/* Buttons */} 
          <Button variant="ghost" size="sm" onClick={toggleScanning}>
            <Shield className={cn("h-4 w-4 mr-1", isScanning ? "text-blue-500" : "text-muted-foreground")} />
            {isScanning ? 'Scanning' : 'Paused'}
          </Button>
          <Button variant="ghost" size="sm" onClick={togglePlay}>
            {isPlaying ? 
              <Square className="h-4 w-4 mr-1 text-muted-foreground" /> : 
              <Play className="h-4 w-4 mr-1 text-green-500" />
            }
            {isPlaying ? 'Pause' : 'Play'}
          </Button>
          <Button variant="ghost" size="sm" onClick={toggleFullscreen}>
            {isFullscreen ? 
              <Minimize2 className="h-4 w-4 mr-1" /> : 
              <Maximize2 className="h-4 w-4 mr-1" />
            }
            {isFullscreen ? 'Exit' : 'Full'}
          </Button>
        </div>
      </div>

      {/* Video/Image Area */} 
      <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden">
        {currentFrameSrc ? (
          <div className="relative w-full h-full">
            <img
              ref={imageRef}
              alt="Video Feed" 
              // src is managed by useEffect to handle load event for dimensions
              className="block w-full h-full object-contain" // Use object-contain to see full frame
              onError={() => setError("Failed to load video frame image.")}
            />

            {/* --- Bounding Box Overlays --- */} 
            {naturalDimensions && imageRef.current && (
              <> 
                {/* Gun Overlays (High Confidence Only) */} 
                {mmdetGuns
                  .filter(gun => gun.confidence > 0.85) // Only include high confidence guns
                  .map((gun, index) => { 
                  const scaleX = imageRef.current!.offsetWidth / naturalDimensions.width;
                  const scaleY = imageRef.current!.offsetHeight / naturalDimensions.height;
                  const left = gun.bbox.xmin * scaleX;
                  const top = gun.bbox.ymin * scaleY;
                  const width = (gun.bbox.xmax - gun.bbox.xmin) * scaleX;
                  const height = (gun.bbox.ymax - gun.bbox.ymin) * scaleY;
                  // Severity is always 'high' here due to filter
                  const borderColorClass = 'border-police-red';

                  return (
                    <div
                      key={`gun-${index}`}
                      // Apply thicker border (e.g., border-4)
                      className={cn(
                        "detection-box absolute border-4 rounded-sm pointer-events-none", 
                        borderColorClass
                      )}
                      style={{ left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` }}
                    >
                      {/* Label for Gun */}
                      <span 
                        className="absolute -top-5 left-0 text-xs font-semibold bg-police-red text-white px-1 py-0.5 rounded-sm"
                        style={{ transform: 'translateY(-100%)' }} // Position above the box
                      >
                        Gun
                      </span>
                    </div>
                  );
                })}

                {/* Person Overlays (Conditional) */} 
                {yoloPersons.map((person, index) => { 
                  const personKey = `Person_${person.id}`;
                  if (personGunState[personKey]) { 
                    const scaleX = imageRef.current!.offsetWidth / naturalDimensions.width;
                    const scaleY = imageRef.current!.offsetHeight / naturalDimensions.height;
                    const [xmin, ymin, xmax, ymax] = person.bbox;
                    const left = xmin * scaleX;
                    const top = ymin * scaleY;
                    const width = (xmax - xmin) * scaleX;
                    const height = (ymax - ymin) * scaleY;

                    return (
                      <div
                        key={`person-${person.id}-${index}`}
                        // Apply thicker border and blue color
                        className="detection-box absolute border-4 rounded-sm border-police-blue pointer-events-none"
                        style={{ left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` }}
                      >
                        {/* Label for Person */}
                        <span 
                          className="absolute -top-5 left-0 text-xs font-semibold bg-police-blue text-white px-1 py-0.5 rounded-sm"
                          style={{ transform: 'translateY(-100%)' }} // Position above the box
                        >
                          {personKey} 
                        </span>
                      </div>
                    );
                  }
                  return null; 
                })}
              </>
            )}
            {/* --- End Overlays --- */}
          </div>
        ) : (
          // Placeholder/Loading/Error State
          <div className="text-muted-foreground text-sm">
            {error ? `Error: ${error}` : 'Connecting to video feed...'}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoFeed;

// Add a placeholder BBox type if not defined elsewhere
// interface BBox { xmin: number; ymin: number; xmax: number; ymax: number; } 