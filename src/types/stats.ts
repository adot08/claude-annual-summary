export interface StatsData {
    conversation_stats: {
      total: number;
      turns: number;
      human_turns: number;
      ai_turns: number;
      avg_turns: number;
      avg_human: number;
      avg_ai: number;
    };
    token_stats: {
      input: number;
      output: number;
      total: number;
    };
    time_stats: {
      busiest_day: {
        date: string;
        count: number;
      };
      busiest_weekday: {
        day: string;
        avg_count: number;
      };
    };
  }