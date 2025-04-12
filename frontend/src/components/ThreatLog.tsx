
import React, { useState } from 'react';
import { AlertTriangle, ArrowDown, ArrowUp, Filter, Shield } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface Threat {
  id: number;
  type: string;
  severity: 'low' | 'medium' | 'high';
  location: string;
  timestamp: Date;
  confidence: number;
}

const ThreatLog = () => {
  const [filter, setFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [sortAsc, setSortAsc] = useState(false);
  
  // Sample threats data
  const [threats] = useState<Threat[]>([
    {
      id: 1,
      type: 'Firearm',
      severity: 'high',
      location: 'Main Entrance',
      timestamp: new Date(Date.now() - 2 * 60 * 1000), // 2 mins ago
      confidence: 0.92,
    },
    {
      id: 2,
      type: 'Knife',
      severity: 'medium',
      location: 'Main Entrance',
      timestamp: new Date(Date.now() - 5 * 60 * 1000), // 5 mins ago
      confidence: 0.85,
    },
    {
      id: 3,
      type: 'Suspicious Behavior',
      severity: 'low',
      location: 'Main Entrance',
      timestamp: new Date(Date.now() - 8 * 60 * 1000), // 8 mins ago
      confidence: 0.78,
    },
    {
      id: 4,
      type: 'Firearm',
      severity: 'high',
      location: 'Main Entrance',
      timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 mins ago
      confidence: 0.88,
    },
  ]);
  
  // Filter and sort threats
  const filteredThreats = threats
    .filter(threat => filter === 'all' ? true : threat.severity === filter)
    .sort((a, b) => {
      if (sortAsc) {
        return a.timestamp.getTime() - b.timestamp.getTime();
      } else {
        return b.timestamp.getTime() - a.timestamp.getTime();
      }
    });
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  const getSeverityColor = (severity: 'low' | 'medium' | 'high') => {
    switch (severity) {
      case 'high': return 'text-alert-high';
      case 'medium': return 'text-alert-medium';
      case 'low': return 'text-alert-low';
      default: return '';
    }
  };
  
  return (
    <div className="h-full flex flex-col bg-card rounded-lg border border-border overflow-hidden">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="h-5 w-5 text-police-red" />
          <h2 className="font-semibold">Threat Log</h2>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSortAsc(!sortAsc)}
            className="h-8 px-2 text-xs"
          >
            {sortAsc ? (
              <ArrowUp className="h-4 w-4 mr-1" />
            ) : (
              <ArrowDown className="h-4 w-4 mr-1" />
            )}
            {sortAsc ? 'Oldest' : 'Newest'}
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setFilter(filter === 'all' ? 'high' : 'all')}
            className="h-8 px-2 text-xs"
          >
            <Filter className="h-4 w-4 mr-1" />
            {filter === 'all' ? 'All' : filter === 'high' ? 'High' : filter === 'medium' ? 'Medium' : 'Low'}
          </Button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        {filteredThreats.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <Shield className="h-12 w-12 mb-2" />
            <p>No threats detected</p>
          </div>
        ) : (
          <ul className="divide-y divide-border">
            {filteredThreats.map((threat) => (
              <li key={threat.id} className={`threat-item ${threat.severity}`}>
                <div className="flex-1">
                  <div className="flex items-center">
                    <h3 className="font-medium">{threat.type}</h3>
                    <span className={cn(
                      "ml-2 px-2 py-0.5 text-xs rounded-full font-medium",
                      threat.severity === 'high' ? 'bg-police-red/20 text-police-red' :
                      threat.severity === 'medium' ? 'bg-alert-medium/20 text-alert-medium' :
                      'bg-alert-low/20 text-alert-low'
                    )}>
                      {threat.severity.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {threat.location} • {formatTime(threat.timestamp)}
                  </div>
                </div>
                <div className={cn(
                  "text-sm font-medium",
                  getSeverityColor(threat.severity)
                )}>
                  {(threat.confidence * 100).toFixed(0)}%
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default ThreatLog;
