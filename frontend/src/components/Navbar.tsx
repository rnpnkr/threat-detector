
import React from 'react';
import { Bell, Settings, Shield } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Navbar = () => {
  return (
    <nav className="flex items-center justify-between px-6 py-4 bg-police-dark border-b border-border">
      <div className="flex items-center space-x-2">
        <Shield className="h-6 w-6 text-police-blue" />
        <span className="text-xl font-bold">PoliceAlertSystem</span>
      </div>
      
      <div className="flex items-center space-x-4">
        <Button variant="ghost" size="icon">
          <Bell className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon">
          <Settings className="h-5 w-5" />
        </Button>
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
            <span className="text-sm font-medium">JD</span>
          </div>
          <span className="text-sm font-medium">Officer Johnson</span>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
