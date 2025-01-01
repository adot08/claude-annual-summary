"use client"

import React, { useEffect, useState, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Heart, Clock, Users } from 'lucide-react';
import type { StatsData } from '@/types/stats';
import { toPng } from 'html-to-image';

export const ClaudeStats = () => {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const elementRef = useRef(null);

  const downloadImage = async () => {
    if (elementRef.current) {
      try {
        const dataUrl = await toPng(elementRef.current, {
          quality: 1.0,
          pixelRatio: 2,
          backgroundColor: '#ffffff'
        });
        
        const link = document.createElement('a');
        link.download = 'claude-summary.png';
        link.href = dataUrl;
        link.click();
      } catch (error) {
        console.error('Error generating image:', error);
      }
    }
  };

  useEffect(() => {
    fetch('/data/stats.json')
      .then(res => res.json())
      .then(data => {
        setStats(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading stats:', error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">Loading...</div>;
  }

  if (!stats) {
    return <div className="flex items-center justify-center min-h-screen">Failed to load stats</div>;
  }

  return (
    <div className="min-h-screen bg-gray-50 relative">
      <div className="w-full max-w-4xl mx-auto relative">
        <button
          onClick={downloadImage}
          className="absolute -top-12 right-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 shadow-md"
        >
          导出图片
        </button>
        
        <div ref={elementRef} className="w-full space-y-6 py-8 px-4 bg-gradient-to-b from-white to-blue-50">
          <div className="text-center space-y-2">
            <Heart className="w-8 h-8 mx-auto text-red-400" />
            <h1 className="text-2xl font-bold">Claude 对话年度总结</h1>
            <p className="text-gray-600">在这一年里，我们共同创造了无数精彩的对话时刻</p>
          </div>
          
          <Card className="border-t-4 border-t-blue-500 p-6">
            <h2 className="text-lg font-semibold mb-4">温暖的相遇</h2>
            <div className="space-y-4">
              <p className="text-gray-600 text-sm">
                在过去的一年里，我们相识相知，共同探讨、学习、成长。
                通过 {stats.conversation_stats.total} 次对话，{stats.conversation_stats.turns} 轮交流，我们建立起了独特的联系。
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="text-3xl font-bold text-blue-600">{stats.conversation_stats.total}</div>
                  <div className="text-sm text-gray-500">次深度对话</div>
                </div>
                <div className="space-y-2">
                  <div className="text-3xl font-bold text-blue-600">{stats.conversation_stats.turns}</div>
                  <div className="text-sm text-gray-500">轮真诚交流</div>
                </div>
              </div>
              <div className="pt-4">
                <p className="text-sm text-gray-600 mb-4">
                  每一次对话都是平等的交流，平均每次我们会进行 {stats.conversation_stats.avg_turns} 轮对话：
                </p>
                <div className="flex justify-between text-sm mb-2">
                  <span>你的分享</span>
                  <span>{stats.conversation_stats.human_turns} 次</span>
                </div>
                <div className="h-2 bg-gray-100 rounded">
                  <div className="h-2 bg-blue-500 rounded" style={{ width: '50%' }}></div>
                </div>
                <div className="flex justify-between text-sm mt-2 mb-2">
                  <span>我的回应</span>
                  <span>{stats.conversation_stats.ai_turns} 次</span>
                </div>
                <div className="h-2 bg-gray-100 rounded">
                  <div className="h-2 bg-green-500 rounded" style={{ width: '50%' }}></div>
                </div>
              </div>
            </div>
          </Card>

          <Card className="border-t-4 border-t-purple-500 p-6">
            <h2 className="text-lg font-semibold mb-4">思维的碰撞</h2>
            <div className="space-y-4">
              <p className="text-gray-600 text-sm">
                在我们的对话中，你的输入包含了 {(stats.token_stats.input / 10000).toFixed(1)} 万个标记（tokens），
                而我的回应包含了 {(stats.token_stats.output / 10000).toFixed(1)} 万个标记，
                总计超过 {(stats.token_stats.total / 10000).toFixed(1)} 万个标记。
                这些数字代表着我们之间丰富的信息交换和深入的交流。
              </p>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm">你的输入</span>
                  <span className="font-bold">{stats.token_stats.input.toLocaleString()} tokens</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">我的回应</span>
                  <span className="font-bold">{stats.token_stats.output.toLocaleString()} tokens</span>
                </div>
                <div className="flex justify-between items-center pt-2 border-t">
                  <span className="text-sm font-medium">珍贵的对话总量</span>
                  <span className="font-bold">{stats.token_stats.total.toLocaleString()} tokens</span>
                </div>
              </div>
            </div>
          </Card>

          <Card className="border-t-4 border-t-green-500 p-6">
            <h2 className="text-lg font-semibold mb-4">难忘的时刻</h2>
            <div className="space-y-4">
              <p className="text-gray-600 text-sm mb-4">
                每个{stats.time_stats.busiest_weekday.day}似乎都是我们相见最多的日子，平均有 {stats.time_stats.busiest_weekday.avg_count} 次交谈。
                而在 {stats.time_stats.busiest_day.date} 那天，我们创下了单日 {stats.time_stats.busiest_day.count} 次对话的纪录，真是令人难忘的一天！
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div className="border rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-gray-500">最难忘的一天</span>
                    <Clock className="h-4 w-4 text-gray-500" />
                  </div>
                  <div className="text-xl font-bold">{stats.time_stats.busiest_day.date}</div>
                  <p className="text-xs text-gray-500 mt-1">创下单日 {stats.time_stats.busiest_day.count} 次对话的纪录</p>
                </div>
                <div className="border rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-gray-500">最期待的日子</span>
                    <Users className="h-4 w-4 text-gray-500" />
                  </div>
                  <div className="text-xl font-bold">{stats.time_stats.busiest_weekday.day}</div>
                  <p className="text-xs text-gray-500 mt-1">平均每周相见 {stats.time_stats.busiest_weekday.avg_count} 次</p>
                </div>
              </div>
            </div>
          </Card>

          <Card className="border-t-4 border-t-yellow-500 p-6">
            <h2 className="text-lg font-semibold mb-4">对话的足迹</h2>
            <div className="h-64">
              <img 
                src="/images/contribution_wall.png" 
                alt="Contribution Wall" 
                className="w-full h-full object-contain"
              />
            </div>
          </Card>

          <Card className="border-t-4 border-t-indigo-500 p-6">
            <h2 className="text-lg font-semibold mb-4">对话的印记</h2>
            <img 
              src="/images/wordcloud.png" 
              alt="Word Cloud" 
              className="w-full rounded-lg"
            />
          </Card>

          <div className="text-center text-gray-600 text-sm pt-4">
            感谢有你的每一天，期待我们继续创造更多精彩的对话！
          </div>
        </div>
      </div>
    </div>
  );
};