import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.db.session import get_db
from src.db.models import Media, Analysis, User
from src.utils.tracing import log_trace
import logging
from enum import Enum

class NotificationType(str, Enum):
    """Types of notifications"""
    ACHIEVEMENT = "achievement"
    IMPROVEMENT = "improvement"
    REMINDER = "reminder"
    SOCIAL = "social"
    MILESTONE = "milestone"
    ALERT = "alert"
    INSIGHT = "insight"
    RECOMMENDATION = "recommendation"

class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class NotificationChannel(str, Enum):
    """Notification delivery channels"""
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"

class SmartNotification(BaseModel):
    """Model for smart notification"""
    id: str = Field(description="Unique notification ID")
    type: NotificationType = Field(description="Type of notification")
    priority: NotificationPriority = Field(description="Priority level")
    title: str = Field(description="Notification title")
    message: str = Field(description="Notification message")
    action_text: Optional[str] = Field(description="Call-to-action text")
    action_url: Optional[str] = Field(description="Action URL")
    channels: List[NotificationChannel] = Field(description="Delivery channels")
    scheduled_time: Optional[str] = Field(description="Scheduled delivery time")
    expires_at: Optional[str] = Field(description="Expiration time")
    metadata: Dict[str, Any] = Field(description="Additional metadata")
    personalization_score: float = Field(description="How personalized this notification is (0-1)")

class NotificationInsight(BaseModel):
    """Model for notification insights"""
    insight_type: str = Field(description="Type of insight")
    title: str = Field(description="Insight title")
    description: str = Field(description="Detailed description")
    confidence: float = Field(description="Confidence in insight (0-1)")
    actionable: bool = Field(description="Whether insight is actionable")
    urgency: str = Field(description="Urgency level")

class NotificationStrategy(BaseModel):
    """Model for notification strategy"""
    user_preferences: Dict[str, Any] = Field(description="User notification preferences")
    optimal_timing: Dict[str, str] = Field(description="Optimal timing for different notification types")
    frequency_limits: Dict[str, int] = Field(description="Frequency limits by type")
    engagement_patterns: Dict[str, float] = Field(description="User engagement patterns")
    personalization_level: float = Field(description="Level of personalization (0-1)")

class NotificationResult(BaseModel):
    """Result model for notification analysis"""
    notifications: List[SmartNotification] = Field(description="Generated notifications")
    insights: List[NotificationInsight] = Field(description="Notification insights")
    strategy: NotificationStrategy = Field(description="Notification strategy")
    engagement_prediction: float = Field(description="Predicted engagement rate (0-1)")
    optimization_suggestions: List[str] = Field(description="Optimization suggestions")
    next_notification_time: Optional[str] = Field(description="Recommended next notification time")
    fatigue_risk: float = Field(description="Risk of notification fatigue (0-1)")
    personalization_opportunities: List[str] = Field(description="Personalization opportunities")
    confidence: float = Field(description="Overall analysis confidence")
    processing_summary: str = Field(description="Summary of notification processing")

class NotificationAgent(BaseAgent):
    """Agent for intelligent notification management and delivery"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Notification parameters
        self.max_daily_notifications = 5
        self.min_notification_interval_hours = 4
        self.engagement_threshold = 0.3
        
        # Timing preferences (hour of day)
        self.optimal_times = {
            NotificationType.ACHIEVEMENT: [9, 12, 18],  # Morning, lunch, evening
            NotificationType.IMPROVEMENT: [10, 15],     # Mid-morning, afternoon
            NotificationType.REMINDER: [8, 20],        # Morning, evening
            NotificationType.SOCIAL: [12, 17, 19],     # Lunch, after work, evening
            NotificationType.MILESTONE: [9, 18],       # Morning, evening
            NotificationType.INSIGHT: [10, 16],        # Mid-morning, afternoon
            NotificationType.RECOMMENDATION: [11, 16]   # Late morning, afternoon
        }
        
        # Priority weights for different triggers
        self.trigger_weights = {
            'score_improvement': 0.8,
            'new_milestone': 0.9,
            'peer_comparison': 0.6,
            'trend_change': 0.7,
            'achievement_unlock': 0.9,
            'reminder_due': 0.5
        }
    
    @log_trace
    def run(self, input_data: AgentInput) -> AgentOutput:
        """Generate intelligent notifications"""
        try:
            # Get user data and analysis history
            user_data = self._get_user_data(input_data.media_id)
            if not user_data:
                return self._create_no_data_response()
            
            analysis_history = self._get_analysis_history(user_data['user_id'])
            
            # Analyze current analysis for notification triggers
            current_analysis = input_data.context.get('analysis_result', {})
            
            # Generate notifications based on triggers
            notifications = self._generate_notifications(user_data, analysis_history, current_analysis)
            
            # Generate insights about notification patterns
            insights = self._generate_insights(user_data, notifications)
            
            # Create notification strategy
            strategy = self._create_notification_strategy(user_data, analysis_history)
            
            # Predict engagement and optimize
            engagement_prediction = self._predict_engagement(notifications, strategy)
            optimization_suggestions = self._generate_optimization_suggestions(notifications, strategy)
            
            # Calculate fatigue risk
            fatigue_risk = self._calculate_fatigue_risk(user_data, notifications)
            
            # Identify personalization opportunities
            personalization_opportunities = self._identify_personalization_opportunities(user_data, notifications)
            
            # Determine next notification time
            next_notification_time = self._calculate_next_notification_time(notifications, strategy)
            
            # Calculate confidence and create summary
            confidence = self._calculate_confidence(notifications, user_data)
            processing_summary = self._create_processing_summary(notifications, insights)
            
            result = NotificationResult(
                notifications=notifications,
                insights=insights,
                strategy=strategy,
                engagement_prediction=engagement_prediction,
                optimization_suggestions=optimization_suggestions,
                next_notification_time=next_notification_time,
                fatigue_risk=fatigue_risk,
                personalization_opportunities=personalization_opportunities,
                confidence=confidence,
                processing_summary=processing_summary
            )
            
            return AgentOutput(
                success=True,
                data=result.dict(),
                confidence=confidence,
                processing_time=0.0,
                agent_name="notification"
            )
            
        except Exception as e:
            self.logger.error(f"Notification analysis failed: {e}")
            return AgentOutput(
                success=False,
                data={},
                confidence=0.0,
                processing_time=0.0,
                agent_name="notification",
                error=str(e)
            )
    
    def _get_user_data(self, media_id: str) -> Optional[Dict[str, Any]]:
        """Get user data"""
        try:
            with get_db() as db:
                media = db.query(Media).filter(Media.id == media_id).first()
                if not media:
                    return None
                
                user = db.query(User).filter(User.id == media.user_id).first()
                if not user:
                    return None
                
                return {
                    'user_id': user.id,
                    'username': getattr(user, 'username', 'User'),
                    'email': getattr(user, 'email', None),
                    'created_at': getattr(user, 'created_at', datetime.now()),
                    'preferences': getattr(user, 'notification_preferences', {})
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get user data: {e}")
            return None
    
    def _get_analysis_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's analysis history"""
        try:
            with get_db() as db:
                analyses = db.query(Analysis).join(Media).filter(
                    Media.user_id == user_id,
                    Analysis.created_at >= datetime.now() - timedelta(days=30)
                ).order_by(Analysis.created_at.desc()).limit(20).all()
                
                history = []
                for analysis in analyses:
                    if analysis.result_data:
                        history.append({
                            'id': analysis.id,
                            'created_at': analysis.created_at,
                            'data': analysis.result_data
                        })
                
                return history
                
        except Exception as e:
            self.logger.error(f"Failed to get analysis history: {e}")
            return []
    
    def _generate_notifications(self, user_data: Dict[str, Any], analysis_history: List[Dict[str, Any]], current_analysis: Dict[str, Any]) -> List[SmartNotification]:
        """Generate notifications based on triggers"""
        notifications = []
        
        # Check for achievements
        achievement_notifications = self._check_achievements(user_data, current_analysis)
        notifications.extend(achievement_notifications)
        
        # Check for improvements
        improvement_notifications = self._check_improvements(analysis_history, current_analysis)
        notifications.extend(improvement_notifications)
        
        # Check for milestones
        milestone_notifications = self._check_milestones(user_data, analysis_history)
        notifications.extend(milestone_notifications)
        
        # Check for social updates
        social_notifications = self._check_social_updates(current_analysis)
        notifications.extend(social_notifications)
        
        # Check for insights
        insight_notifications = self._check_insights(current_analysis)
        notifications.extend(insight_notifications)
        
        # Check for reminders
        reminder_notifications = self._check_reminders(user_data, analysis_history)
        notifications.extend(reminder_notifications)
        
        # Prioritize and limit notifications
        notifications = self._prioritize_notifications(notifications)
        
        return notifications[:self.max_daily_notifications]
    
    def _check_achievements(self, user_data: Dict[str, Any], current_analysis: Dict[str, Any]) -> List[SmartNotification]:
        """Check for achievement-based notifications"""
        notifications = []
        
        overall_score = current_analysis.get('overall_score', 0)
        
        # High score achievement
        if overall_score >= 0.9:
            notifications.append(SmartNotification(
                id=f"achievement_high_score_{datetime.now().timestamp()}",
                type=NotificationType.ACHIEVEMENT,
                priority=NotificationPriority.HIGH,
                title="🎉 Outstanding Performance!",
                message=f"Congratulations! You achieved an exceptional score of {overall_score:.1%}. You're in the top tier!",
                action_text="View Details",
                action_url="/analysis/latest",
                channels=[NotificationChannel.IN_APP, NotificationChannel.PUSH],
                scheduled_time=self._get_optimal_time(NotificationType.ACHIEVEMENT),
                expires_at=(datetime.now() + timedelta(days=7)).isoformat(),
                metadata={"score": overall_score, "achievement_type": "high_score"},
                personalization_score=0.9
            ))
        elif overall_score >= 0.8:
            notifications.append(SmartNotification(
                id=f"achievement_good_score_{datetime.now().timestamp()}",
                type=NotificationType.ACHIEVEMENT,
                priority=NotificationPriority.MEDIUM,
                title="✨ Great Job!",
                message=f"You scored {overall_score:.1%}! You're performing really well.",
                action_text="See Analysis",
                action_url="/analysis/latest",
                channels=[NotificationChannel.IN_APP],
                scheduled_time=self._get_optimal_time(NotificationType.ACHIEVEMENT),
                expires_at=(datetime.now() + timedelta(days=3)).isoformat(),
                metadata={"score": overall_score, "achievement_type": "good_score"},
                personalization_score=0.7
            ))
        
        return notifications
    
    def _check_improvements(self, analysis_history: List[Dict[str, Any]], current_analysis: Dict[str, Any]) -> List[SmartNotification]:
        """Check for improvement-based notifications"""
        notifications = []
        
        if len(analysis_history) < 2:
            return notifications
        
        current_score = current_analysis.get('overall_score', 0)
        previous_score = analysis_history[0]['data'].get('overall_score', 0)
        
        improvement = current_score - previous_score
        
        if improvement >= 0.1:  # Significant improvement
            notifications.append(SmartNotification(
                id=f"improvement_significant_{datetime.now().timestamp()}",
                type=NotificationType.IMPROVEMENT,
                priority=NotificationPriority.HIGH,
                title="📈 Significant Improvement!",
                message=f"Your score improved by {improvement:.1%}! Keep up the great work.",
                action_text="View Progress",
                action_url="/progress",
                channels=[NotificationChannel.IN_APP, NotificationChannel.PUSH],
                scheduled_time=self._get_optimal_time(NotificationType.IMPROVEMENT),
                expires_at=(datetime.now() + timedelta(days=5)).isoformat(),
                metadata={"improvement": improvement, "current_score": current_score},
                personalization_score=0.8
            ))
        elif improvement >= 0.05:  # Moderate improvement
            notifications.append(SmartNotification(
                id=f"improvement_moderate_{datetime.now().timestamp()}",
                type=NotificationType.IMPROVEMENT,
                priority=NotificationPriority.MEDIUM,
                title="👍 Nice Progress!",
                message=f"You're improving! Your score went up by {improvement:.1%}.",
                action_text="Keep Going",
                action_url="/analysis/latest",
                channels=[NotificationChannel.IN_APP],
                scheduled_time=self._get_optimal_time(NotificationType.IMPROVEMENT),
                expires_at=(datetime.now() + timedelta(days=3)).isoformat(),
                metadata={"improvement": improvement, "current_score": current_score},
                personalization_score=0.6
            ))
        
        return notifications
    
    def _check_milestones(self, user_data: Dict[str, Any], analysis_history: List[Dict[str, Any]]) -> List[SmartNotification]:
        """Check for milestone-based notifications"""
        notifications = []
        
        analysis_count = len(analysis_history)
        
        # Analysis count milestones
        milestone_counts = [5, 10, 25, 50, 100]
        for milestone in milestone_counts:
            if analysis_count == milestone:
                notifications.append(SmartNotification(
                    id=f"milestone_analysis_{milestone}_{datetime.now().timestamp()}",
                    type=NotificationType.MILESTONE,
                    priority=NotificationPriority.MEDIUM,
                    title=f"🎯 {milestone} Analyses Milestone!",
                    message=f"You've completed {milestone} analyses! Your dedication is paying off.",
                    action_text="View Journey",
                    action_url="/progress/milestones",
                    channels=[NotificationChannel.IN_APP, NotificationChannel.PUSH],
                    scheduled_time=self._get_optimal_time(NotificationType.MILESTONE),
                    expires_at=(datetime.now() + timedelta(days=7)).isoformat(),
                    metadata={"milestone_type": "analysis_count", "count": milestone},
                    personalization_score=0.8
                ))
                break
        
        return notifications
    
    def _check_social_updates(self, current_analysis: Dict[str, Any]) -> List[SmartNotification]:
        """Check for social-based notifications"""
        notifications = []
        
        # Check if user has high social scores
        individual_results = current_analysis.get('individual_results', {})
        social_result = individual_results.get('social', {})
        
        if isinstance(social_result, dict):
            social_score = social_result.get('score', 0)
            
            if social_score >= 0.85:
                notifications.append(SmartNotification(
                    id=f"social_high_score_{datetime.now().timestamp()}",
                    type=NotificationType.SOCIAL,
                    priority=NotificationPriority.MEDIUM,
                    title="🌟 Social Star!",
                    message="Your social presence is impressive! You're making great connections.",
                    action_text="Share Results",
                    action_url="/share",
                    channels=[NotificationChannel.IN_APP],
                    scheduled_time=self._get_optimal_time(NotificationType.SOCIAL),
                    expires_at=(datetime.now() + timedelta(days=3)).isoformat(),
                    metadata={"social_score": social_score},
                    personalization_score=0.7
                ))
        
        return notifications
    
    def _check_insights(self, current_analysis: Dict[str, Any]) -> List[SmartNotification]:
        """Check for insight-based notifications"""
        notifications = []
        
        # Check for interesting insights in the analysis
        insights = current_analysis.get('insights', [])
        
        high_confidence_insights = [i for i in insights if isinstance(i, dict) and i.get('confidence', 0) > 0.8]
        
        if high_confidence_insights:
            insight = high_confidence_insights[0]  # Take the first high-confidence insight
            notifications.append(SmartNotification(
                id=f"insight_high_confidence_{datetime.now().timestamp()}",
                type=NotificationType.INSIGHT,
                priority=NotificationPriority.MEDIUM,
                title="💡 New Insight Available",
                message=f"We discovered something interesting: {insight.get('title', 'New insight about your analysis')}",
                action_text="Learn More",
                action_url="/insights",
                channels=[NotificationChannel.IN_APP],
                scheduled_time=self._get_optimal_time(NotificationType.INSIGHT),
                expires_at=(datetime.now() + timedelta(days=5)).isoformat(),
                metadata={"insight_type": insight.get('type', 'general')},
                personalization_score=0.8
            ))
        
        return notifications
    
    def _check_reminders(self, user_data: Dict[str, Any], analysis_history: List[Dict[str, Any]]) -> List[SmartNotification]:
        """Check for reminder-based notifications"""
        notifications = []
        
        # Check if user hasn't analyzed in a while
        if analysis_history:
            last_analysis = analysis_history[0]['created_at']
            days_since = (datetime.now() - last_analysis).days
            
            if days_since >= 7:
                notifications.append(SmartNotification(
                    id=f"reminder_comeback_{datetime.now().timestamp()}",
                    type=NotificationType.REMINDER,
                    priority=NotificationPriority.LOW,
                    title="👋 We Miss You!",
                    message=f"It's been {days_since} days since your last analysis. Ready for another check-in?",
                    action_text="Take Analysis",
                    action_url="/analyze",
                    channels=[NotificationChannel.IN_APP, NotificationChannel.EMAIL],
                    scheduled_time=self._get_optimal_time(NotificationType.REMINDER),
                    expires_at=(datetime.now() + timedelta(days=14)).isoformat(),
                    metadata={"days_since": days_since},
                    personalization_score=0.6
                ))
        
        return notifications
    
    def _prioritize_notifications(self, notifications: List[SmartNotification]) -> List[SmartNotification]:
        """Prioritize notifications by importance"""
        priority_order = {
            NotificationPriority.URGENT: 4,
            NotificationPriority.HIGH: 3,
            NotificationPriority.MEDIUM: 2,
            NotificationPriority.LOW: 1
        }
        
        return sorted(notifications, key=lambda n: (
            priority_order.get(n.priority, 0),
            n.personalization_score
        ), reverse=True)
    
    def _get_optimal_time(self, notification_type: NotificationType) -> str:
        """Get optimal time for notification type"""
        optimal_hours = self.optimal_times.get(notification_type, [12])
        
        # Choose the next optimal hour
        current_hour = datetime.now().hour
        next_hour = min([h for h in optimal_hours if h > current_hour], default=optimal_hours[0])
        
        # If no hour today, use first hour tomorrow
        if next_hour <= current_hour:
            target_time = datetime.now().replace(hour=next_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            target_time = datetime.now().replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        return target_time.isoformat()
    
    def _generate_insights(self, user_data: Dict[str, Any], notifications: List[SmartNotification]) -> List[NotificationInsight]:
        """Generate insights about notifications"""
        insights = []
        
        # Analyze notification patterns
        if len(notifications) > 3:
            insights.append(NotificationInsight(
                insight_type="high_activity",
                title="High Notification Activity",
                description="You have multiple notifications today. Consider spacing them out for better engagement.",
                confidence=0.8,
                actionable=True,
                urgency="medium"
            ))
        
        # Analyze notification types
        type_counts = {}
        for notif in notifications:
            type_counts[notif.type] = type_counts.get(notif.type, 0) + 1
        
        if type_counts.get(NotificationType.ACHIEVEMENT, 0) > 1:
            insights.append(NotificationInsight(
                insight_type="achievement_focus",
                title="Achievement-Rich Session",
                description="Multiple achievements detected. Great performance!",
                confidence=0.9,
                actionable=False,
                urgency="low"
            ))
        
        return insights
    
    def _create_notification_strategy(self, user_data: Dict[str, Any], analysis_history: List[Dict[str, Any]]) -> NotificationStrategy:
        """Create notification strategy"""
        # Analyze user preferences (simplified)
        preferences = user_data.get('preferences', {})
        
        # Calculate optimal timing based on history
        optimal_timing = {}
        for notif_type in NotificationType:
            optimal_timing[notif_type.value] = self._get_optimal_time(notif_type)
        
        # Set frequency limits
        frequency_limits = {
            NotificationType.ACHIEVEMENT.value: 3,
            NotificationType.IMPROVEMENT.value: 2,
            NotificationType.REMINDER.value: 1,
            NotificationType.SOCIAL.value: 2,
            NotificationType.MILESTONE.value: 1,
            NotificationType.INSIGHT.value: 2
        }
        
        # Calculate engagement patterns (simplified)
        engagement_patterns = {
            "morning": 0.7,
            "afternoon": 0.8,
            "evening": 0.6
        }
        
        # Calculate personalization level
        personalization_level = 0.8 if len(analysis_history) > 5 else 0.5
        
        return NotificationStrategy(
            user_preferences=preferences,
            optimal_timing=optimal_timing,
            frequency_limits=frequency_limits,
            engagement_patterns=engagement_patterns,
            personalization_level=personalization_level
        )
    
    def _predict_engagement(self, notifications: List[SmartNotification], strategy: NotificationStrategy) -> float:
        """Predict engagement rate for notifications"""
        if not notifications:
            return 0.0
        
        total_score = 0.0
        for notif in notifications:
            # Base engagement from personalization
            base_score = notif.personalization_score * 0.5
            
            # Priority factor
            priority_factor = {
                NotificationPriority.URGENT: 0.9,
                NotificationPriority.HIGH: 0.8,
                NotificationPriority.MEDIUM: 0.6,
                NotificationPriority.LOW: 0.4
            }.get(notif.priority, 0.5)
            
            # Channel factor
            channel_factor = 0.8 if NotificationChannel.PUSH in notif.channels else 0.6
            
            # Timing factor (simplified)
            timing_factor = 0.7
            
            notification_score = (base_score + priority_factor * 0.3 + channel_factor * 0.1 + timing_factor * 0.1)
            total_score += notification_score
        
        return min(total_score / len(notifications), 1.0)
    
    def _generate_optimization_suggestions(self, notifications: List[SmartNotification], strategy: NotificationStrategy) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        # Check notification count
        if len(notifications) > 4:
            suggestions.append("Consider reducing notification frequency to prevent fatigue")
        
        # Check personalization
        avg_personalization = sum(n.personalization_score for n in notifications) / len(notifications) if notifications else 0
        if avg_personalization < 0.6:
            suggestions.append("Increase personalization to improve engagement")
        
        # Check channel diversity
        all_channels = set()
        for notif in notifications:
            all_channels.update(notif.channels)
        
        if len(all_channels) < 2:
            suggestions.append("Diversify notification channels for better reach")
        
        return suggestions
    
    def _calculate_fatigue_risk(self, user_data: Dict[str, Any], notifications: List[SmartNotification]) -> float:
        """Calculate risk of notification fatigue"""
        # Factors affecting fatigue risk
        count_factor = min(len(notifications) / self.max_daily_notifications, 1.0)
        
        # High priority notifications are more fatiguing
        high_priority_count = sum(1 for n in notifications if n.priority in [NotificationPriority.HIGH, NotificationPriority.URGENT])
        priority_factor = min(high_priority_count / 3, 1.0)
        
        # Frequency factor (simplified)
        frequency_factor = 0.3  # Would be calculated from historical data
        
        fatigue_risk = (count_factor * 0.4 + priority_factor * 0.4 + frequency_factor * 0.2)
        
        return min(max(fatigue_risk, 0.0), 1.0)
    
    def _identify_personalization_opportunities(self, user_data: Dict[str, Any], notifications: List[SmartNotification]) -> List[str]:
        """Identify personalization opportunities"""
        opportunities = []
        
        # Check for low personalization scores
        low_personalization = [n for n in notifications if n.personalization_score < 0.6]
        if low_personalization:
            opportunities.append("Enhance personalization for better user engagement")
        
        # Check for generic messaging
        generic_notifications = [n for n in notifications if "You" not in n.message and user_data.get('username', '') not in n.message]
        if len(generic_notifications) > len(notifications) * 0.5:
            opportunities.append("Add more personal touches to notification messages")
        
        # Check for timing optimization
        opportunities.append("Optimize notification timing based on user activity patterns")
        
        return opportunities
    
    def _calculate_next_notification_time(self, notifications: List[SmartNotification], strategy: NotificationStrategy) -> Optional[str]:
        """Calculate recommended next notification time"""
        if not notifications:
            # Default to next day morning
            next_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return next_time.isoformat()
        
        # Find the latest scheduled notification
        latest_time = None
        for notif in notifications:
            if notif.scheduled_time:
                notif_time = datetime.fromisoformat(notif.scheduled_time.replace('Z', '+00:00'))
                if latest_time is None or notif_time > latest_time:
                    latest_time = notif_time
        
        if latest_time:
            # Add minimum interval
            next_time = latest_time + timedelta(hours=self.min_notification_interval_hours)
            return next_time.isoformat()
        
        return None
    
    def _calculate_confidence(self, notifications: List[SmartNotification], user_data: Dict[str, Any]) -> float:
        """Calculate overall confidence"""
        if not notifications:
            return 0.5
        
        # Base confidence from personalization
        avg_personalization = sum(n.personalization_score for n in notifications) / len(notifications)
        
        # Data quality factor
        data_quality = 0.8 if user_data.get('preferences') else 0.6
        
        # Notification quality factor
        quality_factor = min(len(notifications) / 3, 1.0)  # More notifications = higher confidence
        
        confidence = avg_personalization * 0.5 + data_quality * 0.3 + quality_factor * 0.2
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_processing_summary(self, notifications: List[SmartNotification], insights: List[NotificationInsight]) -> str:
        """Create processing summary"""
        if not notifications:
            return "No notifications generated for this analysis."
        
        summary_parts = [
            f"Generated {len(notifications)} notifications",
            f"Identified {len(insights)} insights"
        ]
        
        # Add type breakdown
        type_counts = {}
        for notif in notifications:
            type_counts[notif.type.value] = type_counts.get(notif.type.value, 0) + 1
        
        if type_counts:
            type_summary = ", ".join([f"{count} {type_name}" for type_name, count in type_counts.items()])
            summary_parts.append(f"Types: {type_summary}")
        
        return "; ".join(summary_parts)
    
    def _create_no_data_response(self) -> AgentOutput:
        """Create response when no data is available"""
        result = NotificationResult(
            notifications=[],
            insights=[NotificationInsight(
                insight_type="no_data",
                title="No User Data Available",
                description="Unable to generate personalized notifications without user data.",
                confidence=1.0,
                actionable=True,
                urgency="low"
            )],
            strategy=NotificationStrategy(
                user_preferences={},
                optimal_timing={},
                frequency_limits={},
                engagement_patterns={},
                personalization_level=0.0
            ),
            engagement_prediction=0.0,
            optimization_suggestions=["Collect user preferences for better notifications"],
            next_notification_time=None,
            fatigue_risk=0.0,
            personalization_opportunities=["Set up user preferences and notification settings"],
            confidence=0.0,
            processing_summary="No notifications generated due to insufficient user data"
        )
        
        return AgentOutput(
            success=True,
            data=result.dict(),
            confidence=0.0,
            processing_time=0.0,
            agent_name="notification"
        )
