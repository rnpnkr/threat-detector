
import React, { useState } from 'react';
import Navbar from '@/components/Navbar';
import CameraFeed from '@/components/CameraFeed';
import ThreatLog from '@/components/ThreatLog';
import StatusPanel from '@/components/StatusPanel';
import StatCards from '@/components/StatCards';
import { Separator } from '@/components/ui/separator';

interface Detection {
  id: number;
  type: string;
  confidence: number;
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  severity: 'low' | 'medium' | 'high';
  timestamp: Date;
}

const Index = () => {
  const [detections, setDetections] = useState<Detection[]>([]);

  const handleDetection = (detection: Detection) => {
    setDetections(prev => [detection, ...prev].slice(0, 10));
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      
      <main className="flex-1 p-6 space-y-6">
        <h1 className="text-2xl font-bold">Threat Detection Dashboard</h1>
        
        <StatCards />
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[500px]">
          <div className="lg:col-span-2 h-full">
            <CameraFeed onDetection={handleDetection} />
          </div>
          
          <div className="h-full">
            <ThreatLog />
          </div>
        </div>
        
        <Separator />
        
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">System Status</h2>
          <StatusPanel />
        </div>
      </main>
    </div>
  );
};

export default Index;
