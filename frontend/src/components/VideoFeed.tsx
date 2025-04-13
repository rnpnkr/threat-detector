import React, { useState, useEffect, useRef } from 'react';
import { Camera, Maximize2, Minimize2, Play, RefreshCw, Shield, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface Detection {
  id: number;
  type: string;
  confidence: number;
  severity: 'low' | 'medium' | 'high';
  timestamp: Date;
  bbox: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
}

interface VideoFrame {
  frame: string;
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
  frame_number: number;
  total_frames: number;
}

const VideoFeed = ({ onDetection }: { onDetection: (detection: Detection) => void }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isScanning, setIsScanning] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    // Initialize WebSocket connection for video
    const socket = new WebSocket('ws://localhost:5001/ws/video');
    
    socket.onopen = () => {
      console.log('Video WebSocket connection established');
      setIsPlaying(true);
      setError(null);
    };
    
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'video_frame') {
          const frameData: VideoFrame = data.payload;
          
          // Create a new image element
          const img = new Image();
          img.onload = () => {
            if (canvasRef.current) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx) {
                // Clear canvas
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                
                // Draw the frame
                canvasRef.current.width = img.width;
                canvasRef.current.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                // Draw detection boxes
                frameData.detections.forEach(detection => {
                  const { xmin, ymin, xmax, ymax } = detection.bbox;
                  const severity = detection.confidence > 0.7 ? 'high' : 
                                 detection.confidence > 0.4 ? 'medium' : 'low';
                  
                  // Draw box
                  ctx.strokeStyle = severity === 'high' ? '#ef4444' : 
                                  severity === 'medium' ? '#f59e0b' : '#10b981';
                  ctx.lineWidth = 2;
                  ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
                  
                  // Draw label
                  ctx.fillStyle = severity === 'high' ? '#ef4444' : 
                                severity === 'medium' ? '#f59e0b' : '#10b981';
                  ctx.font = '14px Arial';
                  ctx.fillText(
                    `${detection.class} ${(detection.confidence * 100).toFixed(0)}%`,
                    xmin,
                    ymin - 5
                  );
                });
              }
            }
          };
          img.src = `data:image/jpeg;base64,${frameData.frame}`;
          
          // Process detections
          frameData.detections.forEach(detection => {
            const processedDetection: Detection = {
              id: Date.now(),
              type: detection.class,
              confidence: detection.confidence,
              severity: detection.confidence > 0.7 ? 'high' : detection.confidence > 0.4 ? 'medium' : 'low',
              timestamp: new Date(),
              bbox: detection.bbox
            };
            
            onDetection(processedDetection);
            
            // Show alert for high severity detections
            if (processedDetection.severity === 'high') {
              toast({
                title: `ALERT: ${processedDetection.type} detected`,
                description: `Confidence: ${(processedDetection.confidence * 100).toFixed(0)}%`,
                variant: "destructive"
              });
            }
          });
        }
      } catch (error) {
        console.error('Error processing video frame:', error);
        setError('Error processing video frame');
      }
    };
    
    socket.onerror = (error) => {
      console.error('Video WebSocket error:', error);
      setError('WebSocket connection error');
    };
    
    socket.onclose = () => {
      console.log('Video WebSocket connection closed');
      setIsPlaying(false);
      setError('WebSocket connection closed');
    };
    
    setWs(socket);
    
    return () => {
      socket.close();
    };
  }, []);

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
    toast({
      title: isPlaying ? "Video paused" : "Video playing",
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
          <h2 className="font-semibold">Video Feed</h2>
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
            onClick={togglePlay}
            className="h-8 px-2 text-xs"
          >
            {isPlaying ? (
              <Play className="h-4 w-4 mr-1 text-police-blue" />
            ) : (
              <Square className="h-4 w-4 mr-1 text-muted-foreground" />
            )}
            {isPlaying ? 'Playing' : 'Paused'}
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
      
      <div className="video-feed flex-1">
        <canvas 
          ref={canvasRef}
          className="w-full h-full object-cover"
        />
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <div className="bg-background p-4 rounded-lg shadow-lg">
              <p className="text-destructive">{error}</p>
            </div>
          </div>
        )}
        
        <div className="video-overlay">
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

export default VideoFeed; 