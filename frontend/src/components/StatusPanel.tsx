
import React from 'react';
import { ActivitySquare, Bell, Camera, Check, Clock, Shield, X } from 'lucide-react';
import { cn } from '@/lib/utils';

const StatusItem = ({ 
  icon: Icon, 
  label, 
  value, 
  status 
}: { 
  icon: React.ElementType; 
  label: string; 
  value: string; 
  status: 'online' | 'offline' | 'warning' 
}) => {
  return (
    <div className="flex items-center space-x-3 p-3 bg-card rounded-md border border-border">
      <div className={cn(
        "p-2 rounded-md",
        status === 'online' ? 'bg-green-500/10' : 
        status === 'warning' ? 'bg-yellow-500/10' : 
        'bg-red-500/10'
      )}>
        <Icon className={cn(
          "h-5 w-5",
          status === 'online' ? 'text-green-500' : 
          status === 'warning' ? 'text-yellow-500' : 
          'text-red-500'
        )} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs text-muted-foreground truncate">{value}</p>
      </div>
      <div className={cn(
        "w-2 h-2 rounded-full",
        status === 'online' ? 'bg-green-500' : 
        status === 'warning' ? 'bg-yellow-500' : 
        'bg-red-500'
      )} />
    </div>
  );
};

const StatusPanel = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      <StatusItem
        icon={Camera}
        label="Camera System"
        value="4 cameras active, 0 offline"
        status="online"
      />
      <StatusItem
        icon={Shield}
        label="Detection Engine"
        value="Running v2.4.1"
        status="online"
      />
      <StatusItem
        icon={Bell}
        label="Alert System"
        value="Connected to dispatch"
        status="online"
      />
      <StatusItem
        icon={ActivitySquare}
        label="System Status"
        value="All systems operational"
        status="online"
      />
    </div>
  );
};

export default StatusPanel;
