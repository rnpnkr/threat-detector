import React, { useState, useEffect } from 'react';
import { Camera, Maximize2, Minimize2, Play, RefreshCw, Shield, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { config } from '@/config';

interface Detection {
  id: number;
  type: string;
  confidence: number;
  severity: 'low' | 'medium' | 'high';
  timestamp: Date;
}

interface BackendResponse {
  message: string;
  detections: Array<{
    bbox: {
      xmin: number;
      ymin: number;
      xmax: number;
      ymax: number;
    };
    class: string;
    confidence: number;
  }>;
  image_info: {
    shape: number[];
  };
  original_image: string;
  annotated_image: string;
}

const CameraFeed = ({ onDetection }: { onDetection: (detection: Detection) => void }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isRecording, setIsRecording] = useState(true);
  const [isScanning, setIsScanning] = useState(true);
  const [cameraFeedUrl, setCameraFeedUrl] = useState('https://placehold.co/1280x720/333/777?text=Camera+Feed');
  const [ws, setWs] = useState<WebSocket | null>(null);

  const processBackendDetection = (
    detection: BackendResponse['detections'][0]
  ): Detection => {
    return {
      id: Date.now(),
      type: detection.class,
      confidence: detection.confidence,
      severity: detection.confidence > 0.7 ? 'high' : detection.confidence > 0.4 ? 'medium' : 'low',
      timestamp: new Date(),
    };
  };

  const handleNewImage = (data: BackendResponse) => {
    console.log('New image detected:', data);
    
    // Update camera feed with annotated image using the config helper
    if (data.annotated_image) {
        const imageUrl = config.getStaticImageUrl(data.annotated_image);
        console.log(`Setting CameraFeed image URL to: ${imageUrl}`);
        setCameraFeedUrl(imageUrl);
    } else {
        console.warn("Received new_detection payload without an annotated_image path.");
        // Optionally set a default/placeholder image
        // setCameraFeedUrl('https://placehold.co/1280x720/333/777?text=No+Annotated+Image');
    }
    
    // Process detections for alerts only
    const newDetections = data.detections.map(detection => processBackendDetection(detection));
    
    // Show alert for high severity detections
    newDetections.forEach(detection => {
      if (detection.severity === 'high') {
        console.log('High severity detection found:', detection);
        toast({
          title: `ALERT: ${detection.type} detected`,
          description: `Confidence: ${(detection.confidence * 100).toFixed(0)}%`,
          variant: "destructive"
        });
      }
    });

    // Notify parent component of detections
    newDetections.forEach(detection => onDetection(detection));
  };

  useEffect(() => {
    // Initialize WebSocket connection
    // const socket = new WebSocket('ws://localhost:5001/ws'); // OLD notification socket
    // NOTE: This component seems to listen for 'new_detection' which was tied to the /detect endpoint.
    // It might need updating to use the new video feed logic or be deprecated if video feed handles all updates.
    // For now, commenting out the old connection.
    /*
    const socket = new WebSocket('ws://localhost:5001/ws'); // Replace with appropriate new logic if needed

    socket.onopen = () => {
      console.log('WebSocket connection established');
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'new_detection') {
          handleNewImage(data.payload);
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    socket.onclose = () => {
      console.log('WebSocket connection closed');
    };

    setWs(socket);

    return () => {
      socket.close();
    };
    */
  }, []);

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    toast({
      title: isRecording ? "Recording paused" : "Recording resumed",
      variant: "default"
    });
  };

  const toggleScanning = () => {
    setIsScanning(!isScanning);
    toast({
      title: isScanning ? "Threat detection paused" : "Threat detection activated",
      variant: "default"
    });
  };

  return (
    <div className={cn(
      "flex flex-col", 
      isFullscreen ? "fixed inset-0 z-50 p-4 bg-background" : "h-full"
    )}>
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center space-x-2">
          <Camera className="h-5 w-5 text-police-blue" />
          <h2 className="font-semibold">Camera Feed</h2>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleScanning}
            className="h-8 px-2 text-xs"
          >
            {isScanning ? (
              <Shield className="h-4 w-4 mr-1 text-police-blue" />
            ) : (
              <Shield className="h-4 w-4 mr-1 text-muted-foreground" />
            )}
            {isScanning ? 'Scanning' : 'Paused'}
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleRecording}
            className="h-8 px-2 text-xs"
          >
            {isRecording ? (
              <Play className="h-4 w-4 mr-1 text-police-blue" />
            ) : (
              <Square className="h-4 w-4 mr-1 text-muted-foreground" />
            )}
            {isRecording ? 'Recording' : 'Paused'}
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleFullscreen}
            className="h-8 px-2 text-xs"
          >
            {isFullscreen ? (
              <Minimize2 className="h-4 w-4 mr-1" />
            ) : (
              <Maximize2 className="h-4 w-4 mr-1" />
            )}
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </Button>
        </div>
      </div>
      
      <div className="camera-feed flex-1">
        <img 
          src={cameraFeedUrl} 
          alt="Camera Feed" 
          className="w-full h-full object-cover"
        />
        
        <div className="camera-overlay">
          {/* Scanning indicator */}
          {isScanning && (
            <div className="absolute bottom-4 right-4 flex items-center space-x-2 bg-background/70 text-foreground px-3 py-1 rounded-full text-xs">
              <RefreshCw className="h-3 w-3 animate-rotate-scan text-police-blue" />
              <span>Scanning for threats</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CameraFeed;
