
import React from 'react';
import { AlarmClockOff, AlertTriangle, Clock, ShieldAlert } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

const StatCards = () => {
  // For a demo we'll use static data, in a real app these would be fetched
  const stats = [
    {
      title: 'Threats Detected Today',
      value: '4',
      description: '2 high-priority',
      icon: AlertTriangle,
      iconColor: 'text-police-red',
      change: '+2 from yesterday',
      trend: 'up',
    },
    {
      title: 'Average Response Time',
      value: '1m 24s',
      description: 'to high-priority threats',
      icon: Clock,
      iconColor: 'text-alert-medium',
      change: '-12s from average',
      trend: 'down',
    },
    {
      title: 'System Uptime',
      value: '99.8%',
      description: 'last 30 days',
      icon: ShieldAlert,
      iconColor: 'text-police-blue',
      change: '+0.2% from last month',
      trend: 'up',
    },
    {
      title: 'Offline Duration',
      value: '12m',
      description: 'total this month',
      icon: AlarmClockOff,
      iconColor: 'text-muted-foreground',
      change: '-35m from last month',
      trend: 'down',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <Card key={index}>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">
              {stat.title}
            </CardTitle>
            <stat.icon className={`h-4 w-4 ${stat.iconColor}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stat.value}</div>
            <CardDescription className="text-xs">{stat.description}</CardDescription>
            <p className={`text-xs mt-1 ${stat.trend === 'up' ? 'text-police-red' : 'text-green-500'}`}>
              {stat.change}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default StatCards;
