# project_management_system/
"""
نظام متكامل لإدارة المشاريع وتحليل الإنتاجية
يشمل:
- إدارة المهام والمشاريع
- تحليل أداء الفريق
- تتبع الوقت والإنتاجية
- تقارير الأداء
- تحليل المخاطر
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass
from typing import Any
import warnings
warnings.filterwarnings('ignore')

# ============= Enums and Data Classes =============
class TaskStatus(Enum):
    TODO = "للبدء"
    IN_PROGRESS = "قيد التنفيذ"
    REVIEW = "مراجعة"
    DONE = "مكتمل"
    BLOCKED = "متوقف"

class TaskPriority(Enum):
    LOW = "منخفض"
    MEDIUM = "متوسط"
    HIGH = "عالي"
    CRITICAL = "حرج"

class ProjectStatus(Enum):
    PLANNING = "تخطيط"
    ACTIVE = "نشط"
    ON_HOLD = "معلق"
    COMPLETED = "مكتمل"
    CANCELLED = "ملغي"

@dataclass
class Task:
    id: str
    title: str
    description: str
    project_id: str
    assigned_to: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    due_date: datetime
    completed_at: Optional[datetime]
    estimated_hours: float
    actual_hours: float
    dependencies: List[str]
    tags: List[str]
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'project_id': self.project_id,
            'assigned_to': self.assigned_to,
            'status': self.status.value,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'due_date': self.due_date.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'estimated_hours': self.estimated_hours,
            'actual_hours': self.actual_hours,
            'dependencies': self.dependencies,
            'tags': self.tags
        }

@dataclass
class Project:
    id: str
    name: str
    description: str
    manager: str
    status: ProjectStatus
    start_date: datetime
    end_date: Optional[datetime]
    budget: float
    spent: float
    team_members: List[str]
    tasks: List[str]
    risks: List[Dict]
    milestones: List[Dict]
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'manager': self.manager,
            'status': self.status.value,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'budget': self.budget,
            'spent': self.spent,
            'team_members': self.team_members,
            'tasks': self.tasks,
            'risks': self.risks,
            'milestones': self.milestones
        }

@dataclass
class TeamMember:
    id: str
    name: str
    email: str
    role: str
    skills: List[str]
    current_projects: List[str]
    workload: float
    performance_score: float
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'skills': self.skills,
            'current_projects': self.current_projects,
            'workload': self.workload,
            'performance_score': self.performance_score
        }

# ============= Main Project Management System =============
class ProjectManagementSystem:
    """
    نظام متكامل لإدارة المشاريع وتحليل الإنتاجية
    """
    
    def __init__(self):
        self.projects = {}
        self.tasks = {}
        self.team_members = {}
        self.activity_log = []
        self.setup_logging()
        
    def setup_logging(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('project_management.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    # ============= Project Management Module =============
    def create_project(self, name: str, description: str, manager: str, 
                      start_date: datetime, budget: float) -> str:
        """إنشاء مشروع جديد"""
        project_id = self.generate_id('PROJ')
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            manager=manager,
            status=ProjectStatus.PLANNING,
            start_date=start_date,
            end_date=None,
            budget=budget,
            spent=0.0,
            team_members=[manager],
            tasks=[],
            risks=[],
            milestones=[]
        )
        
        self.projects[project_id] = project
        self.log_activity('CREATE_PROJECT', {'project_id': project_id, 'name': name})
        self.logger.info(f"تم إنشاء مشروع جديد: {name} (ID: {project_id})")
        
        return project_id
    
    def update_project_status(self, project_id: str, status: ProjectStatus):
        """تحديث حالة المشروع"""
        if project_id in self.projects:
            self.projects[project_id].status = status
            self.log_activity('UPDATE_PROJECT_STATUS', {
                'project_id': project_id,
                'new_status': status.value
            })
            return True
        return False
    
    def add_team_member_to_project(self, project_id: str, member_id: str):
        """إضافة عضو فريق إلى المشروع"""
        if project_id in self.projects and member_id in self.team_members:
            if member_id not in self.projects[project_id].team_members:
                self.projects[project_id].team_members.append(member_id)
                self.team_members[member_id].current_projects.append(project_id)
                self.log_activity('ADD_TEAM_MEMBER', {
                    'project_id': project_id,
                    'member_id': member_id
                })
                return True
        return False
    
    # ============= Task Management Module =============
    def create_task(self, title: str, description: str, project_id: str,
                   assigned_to: str, priority: TaskPriority, due_date: datetime,
                   estimated_hours: float, dependencies: List[str] = None) -> str:
        """إنشاء مهمة جديدة"""
        task_id = self.generate_id('TASK')
        
        task = Task(
            id=task_id,
            title=title,
            description=description,
            project_id=project_id,
            assigned_to=assigned_to,
            status=TaskStatus.TODO,
            priority=priority,
            created_at=datetime.now(),
            due_date=due_date,
            completed_at=None,
            estimated_hours=estimated_hours,
            actual_hours=0.0,
            dependencies=dependencies or [],
            tags=[]
        )
        
        self.tasks[task_id] = task
        
        # إضافة المهمة إلى المشروع
        if project_id in self.projects:
            self.projects[project_id].tasks.append(task_id)
        
        self.log_activity('CREATE_TASK', {
            'task_id': task_id,
            'project_id': project_id,
            'assigned_to': assigned_to
        })
        
        return task_id
    
    def update_task_status(self, task_id: str, status: TaskStatus):
        """تحديث حالة المهمة"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            old_status = task.status
            task.status = status
            
            if status == TaskStatus.DONE and not task.completed_at:
                task.completed_at = datetime.now()
            
            self.log_activity('UPDATE_TASK_STATUS', {
                'task_id': task_id,
                'old_status': old_status.value,
                'new_status': status.value
            })
            return True
        return False
    
    def add_task_dependency(self, task_id: str, depends_on: str):
        """إضافة تبعية بين المهام"""
        if task_id in self.tasks and depends_on in self.tasks:
            if depends_on not in self.tasks[task_id].dependencies:
                self.tasks[task_id].dependencies.append(depends_on)
                self.log_activity('ADD_DEPENDENCY', {
                    'task_id': task_id,
                    'depends_on': depends_on
                })
                return True
        return False
    
    def log_task_hours(self, task_id: str, hours: float):
        """تسجيل ساعات العمل على مهمة"""
        if task_id in self.tasks:
            self.tasks[task_id].actual_hours += hours
            
            # تحديث المصروفات في المشروع
            project_id = self.tasks[task_id].project_id
            if project_id in self.projects:
                # حساب التكلفة (افتراضي 100 ريال/ساعة)
                cost = hours * 100
                self.projects[project_id].spent += cost
            
            self.log_activity('LOG_HOURS', {
                'task_id': task_id,
                'hours': hours
            })
            return True
        return False
    
    # ============= Team Management Module =============
    def add_team_member(self, name: str, email: str, role: str, skills: List[str]) -> str:
        """إضافة عضو فريق جديد"""
        member_id = self.generate_id('MEM')
        
        member = TeamMember(
            id=member_id,
            name=name,
            email=email,
            role=role,
            skills=skills,
            current_projects=[],
            workload=0.0,
            performance_score=100.0
        )
        
        self.team_members[member_id] = member
        self.log_activity('ADD_TEAM_MEMBER', {'member_id': member_id, 'name': name})
        
        return member_id
    
    def update_member_workload(self, member_id: str):
        """تحديث عبء العمل لعضو الفريق"""
        if member_id not in self.team_members:
            return
        
        member = self.team_members[member_id]
        
        # حساب عدد المهام النشطة
        active_tasks = sum(1 for task in self.tasks.values()
                          if task.assigned_to == member_id
                          and task.status in [TaskStatus.IN_PROGRESS, TaskStatus.REVIEW])
        
        # حساب الساعات المتبقية
        remaining_hours = sum(task.estimated_hours - task.actual_hours
                            for task in self.tasks.values()
                            if task.assigned_to == member_id
                            and task.status != TaskStatus.DONE)
        
        member.workload = (active_tasks * 10) + (remaining_hours * 0.5)
    
    def calculate_member_performance(self, member_id: str) -> float:
        """حساب أداء عضو الفريق"""
        if member_id not in self.team_members:
            return 0.0
        
        member_tasks = [task for task in self.tasks.values()
                       if task.assigned_to == member_id]
        
        if not member_tasks:
            return 100.0
        
        # مقاييس الأداء
        on_time_completion = 0
        estimation_accuracy = 0
        completed_tasks = 0
        
        for task in member_tasks:
            if task.status == TaskStatus.DONE and task.completed_at:
                completed_tasks += 1
                
                # التسليم في الوقت المحدد
                if task.completed_at <= task.due_date:
                    on_time_completion += 1
                
                # دقة التقدير
                if task.estimated_hours > 0:
                    accuracy = 1 - abs(task.actual_hours - task.estimated_hours) / task.estimated_hours
                    estimation_accuracy += max(0, accuracy)
        
        performance_score = 100.0
        
        if completed_tasks > 0:
            on_time_rate = (on_time_completion / completed_tasks) * 100
            accuracy_rate = (estimation_accuracy / completed_tasks) * 100
            
            performance_score = (on_time_rate * 0.6) + (accuracy_rate * 0.4)
        
        self.team_members[member_id].performance_score = performance_score
        return performance_score
    
    # ============= Risk Management Module =============
    def add_risk(self, project_id: str, risk_description: str, 
                 probability: float, impact: float, mitigation: str) -> dict:
        """إضافة مخاطر للمشروع"""
        risk = {
            'id': self.generate_id('RISK'),
            'description': risk_description,
            'probability': probability,
            'impact': impact,
            'score': probability * impact,
            'mitigation': mitigation,
            'status': 'identified',
            'created_at': datetime.now().isoformat()
        }
        
        if project_id in self.projects:
            self.projects[project_id].risks.append(risk)
            
        return risk
    
    def analyze_project_risks(self, project_id: str) -> dict:
        """تحليل مخاطر المشروع"""
        if project_id not in self.projects:
            return {}
        
        risks = self.projects[project_id].risks
        
        analysis = {
            'total_risks': len(risks),
            'high_risks': sum(1 for r in risks if r['score'] > 0.7),
            'medium_risks': sum(1 for r in risks if 0.3 <= r['score'] <= 0.7),
            'low_risks': sum(1 for r in risks if r['score'] < 0.3),
            'risk_score': sum(r['score'] for r in risks) / len(risks) if risks else 0,
            'top_risks': sorted(risks, key=lambda x: x['score'], reverse=True)[:3]
        }
        
        return analysis
    
    # ============= Analytics Module =============
    def analyze_project_progress(self, project_id: str) -> dict:
        """تحليل تقدم المشروع"""
        if project_id not in self.projects:
            return {}
        
        project = self.projects[project_id]
        project_tasks = [task for task in self.tasks.values()
                        if task.project_id == project_id]
        
        if not project_tasks:
            return {'progress': 0, 'status': 'لا توجد مهام'}
        
        total_tasks = len(project_tasks)
        completed_tasks = sum(1 for task in project_tasks
                            if task.status == TaskStatus.DONE)
        in_progress_tasks = sum(1 for task in project_tasks
                               if task.status == TaskStatus.IN_PROGRESS)
        blocked_tasks = sum(1 for task in project_tasks
                           if task.status == TaskStatus.BLOCKED)
        
        # حساب التقدم
        progress = (completed_tasks / total_tasks) * 100
        
        # حساب الوقت المستغرق
        total_estimated = sum(task.estimated_hours for task in project_tasks)
        total_actual = sum(task.actual_hours for task in project_tasks)
        
        time_variance = ((total_actual - total_estimated) / total_estimated) * 100 if total_estimated > 0 else 0
        
        # حساب الميزانية
        budget_variance = ((project.spent - project.budget) / project.budget) * 100 if project.budget > 0 else 0
        
        analysis = {
            'progress': progress,
            'completed_tasks': completed_tasks,
            'total_tasks': total_tasks,
            'in_progress_tasks': in_progress_tasks,
            'blocked_tasks': blocked_tasks,
            'completion_rate': (completed_tasks / total_tasks) * 100,
            'time_variance': time_variance,
            'budget_variance': budget_variance,
            'estimated_hours': total_estimated,
            'actual_hours': total_actual,
            'remaining_hours': total_estimated - total_actual,
            'budget_used': project.spent,
            'budget_remaining': project.budget - project.spent,
            'days_remaining': (project.end_date - datetime.now()).days if project.end_date else None
        }
        
        return analysis
    
    def analyze_team_productivity(self) -> dict:
        """تحليل إنتاجية الفريق"""
        if not self.team_members:
            return {}
        
        analysis = {
            'total_members': len(self.team_members),
            'avg_performance': np.mean([m.performance_score for m in self.team_members.values()]),
            'avg_workload': np.mean([m.workload for m in self.team_members.values()]),
            'members': {}
        }
        
        for member_id, member in self.team_members.items():
            member_tasks = [task for task in self.tasks.values()
                          if task.assigned_to == member_id]
            
            analysis['members'][member_id] = {
                'name': member.name,
                'role': member.role,
                'performance': member.performance_score,
                'workload': member.workload,
                'active_tasks': sum(1 for task in member_tasks
                                   if task.status == TaskStatus.IN_PROGRESS),
                'completed_tasks': sum(1 for task in member_tasks
                                      if task.status == TaskStatus.DONE),
                'total_hours': sum(task.actual_hours for task in member_tasks)
            }
        
        return analysis
    
    def calculate_critical_path(self, project_id: str) -> List[str]:
        """حساب المسار الحرج للمشروع"""
        project_tasks = [task for task in self.tasks.values()
                        if task.project_id == project_id]
        
        # بناء الرسم البياني للتبعيات
        G = nx.DiGraph()
        
        for task in project_tasks:
            G.add_node(task.id, duration=task.estimated_hours)
            for dep in task.dependencies:
                if dep in self.tasks:
                    G.add_edge(dep, task.id)
        
        # حساب المسار الحرج (أطول مسار)
        try:
            critical_path = nx.dag_longest_path(G, weight='duration')
            return critical_path
        except:
            return []
    
    def predict_completion_date(self, project_id: str) -> datetime:
        """التنبؤ بتاريخ اكتمال المشروع"""
        analysis = self.analyze_project_progress(project_id)
        
        if 'remaining_hours' not in analysis or analysis['remaining_hours'] <= 0:
            return datetime.now()
        
        # حساب متوسط الإنتاجية اليومية
        project_tasks = [task for task in self.tasks.values()
                        if task.project_id == project_id]
        
        completed_tasks = [task for task in project_tasks
                          if task.status == TaskStatus.DONE]
        
        if not completed_tasks:
            return None
        
        # متوسط الساعات المنجزة في اليوم
        avg_daily_hours = np.mean([task.actual_hours / 
                                   max(1, (task.completed_at - task.created_at).days)
                                   for task in completed_tasks if task.completed_at])
        
        days_needed = analysis['remaining_hours'] / max(avg_daily_hours, 1)
        
        return datetime.now() + timedelta(days=days_needed)
    
    # ============= Reporting Module =============
    def generate_project_report(self, project_id: str) -> dict:
        """توليد تقرير المشروع"""
        if project_id not in self.projects:
            return {}
        
        project = self.projects[project_id]
        progress = self.analyze_project_progress(project_id)
        risks = self.analyze_project_risks(project_id)
        
        # حساب مؤشرات الأداء الرئيسية
        kpis = {
            'schedule_performance_index': self.calculate_spi(project_id),
            'cost_performance_index': self.calculate_cpi(project_id),
            'estimate_at_completion': self.calculate_eac(project_id),
            'to_complete_performance_index': self.calculate_tcpi(project_id)
        }
        
        report = {
            'project_info': {
                'id': project.id,
                'name': project.name,
                'manager': project.manager,
                'status': project.status.value,
                'start_date': project.start_date.isoformat(),
                'end_date': project.end_date.isoformat() if project.end_date else None,
                'budget': project.budget,
                'spent': project.spent
            },
            'progress': progress,
            'risks': risks,
            'kpis': kpis,
            'team_performance': {},
            'recommendations': []
        }
        
        # أداء الفريق في المشروع
        for member_id in project.team_members:
            if member_id in self.team_members:
                member = self.team_members[member_id]
                member_tasks = [task for task in self.tasks.values()
                              if task.project_id == project_id 
                              and task.assigned_to == member_id]
                
                report['team_performance'][member_id] = {
                    'name': member.name,
                    'role': member.role,
                    'tasks_completed': sum(1 for task in member_tasks
                                          if task.status == TaskStatus.DONE),
                    'tasks_in_progress': sum(1 for task in member_tasks
                                            if task.status == TaskStatus.IN_PROGRESS),
                    'total_hours': sum(task.actual_hours for task in member_tasks)
                }
        
        # توصيات
        report['recommendations'] = self.generate_recommendations(project_id, progress, risks)
        
        return report
    
    def calculate_spi(self, project_id: str) -> float:
        """حساب مؤشر أداء الجدول الزمني"""
        project_tasks = [task for task in self.tasks.values()
                        if task.project_id == project_id]
        
        if not project_tasks:
            return 1.0
        
        # القيمة المكتسبة (EV) - بناءً على المهام المنجزة
        earned_value = sum(task.estimated_hours for task in project_tasks
                          if task.status == TaskStatus.DONE)
        
        # القيمة المخططة (PV)
        planned_value = sum(task.estimated_hours for task in project_tasks)
        
        return earned_value / planned_value if planned_value > 0 else 1.0
    
    def calculate_cpi(self, project_id: str) -> float:
        """حساب مؤشر أداء التكلفة"""
        project_tasks = [task for task in self.tasks.values()
                        if task.project_id == project_id]
        
        if not project_tasks:
            return 1.0
        
        # القيمة المكتسبة (EV)
        earned_value = sum(task.estimated_hours for task in project_tasks
                          if task.status == TaskStatus.DONE)
        
        # التكلفة الفعلية (AC)
        actual_cost = sum(task.actual_hours for task in project_tasks)
        
        return earned_value / actual_cost if actual_cost > 0 else 1.0
    
    def calculate_eac(self, project_id: str) -> float:
        """حساب التكلفة التقديرية عند الاكتمال"""
        cpi = self.calculate_cpi(project_id)
        project = self.projects[project_id]
        
        if cpi > 0:
            return project.spent + (project.budget - project.spent) / cpi
        return project.budget
    
    def calculate_tcpi(self, project_id: str) -> float:
        """حساب مؤشر الأداء المطلوب للاكتمال"""
        project = self.projects[project_id]
        eac = self.calculate_eac(project_id)
        
        remaining_work = project.budget - project.spent
        remaining_funds = eac - project.spent
        
        return remaining_work / remaining_funds if remaining_funds > 0 else 1.0
    
    def generate_recommendations(self, project_id: str, progress: dict, risks: dict) -> List[str]:
        """توليد توصيات للمشروع"""
        recommendations = []
        
        # توصيات بناءً على التقدم
        if progress['progress'] < 30 and progress['time_variance'] > 20:
            recommendations.append("تأخر في الجدول الزمني - ينصح بإعادة تخطيط المهام المتبقية")
        
        if progress['blocked_tasks'] > 0:
            recommendations.append(f"هناك {progress['blocked_tasks']} مهام متوقفة - تحتاج إلى تدخل فوري")
        
        # توصيات بناءً على الميزانية
        if progress['budget_variance'] > 10:
            recommendations.append("تجاوز في الميزانية - ينصح بمراجعة المصروفات")
        
        # توصيات بناءً على المخاطر
        if risks.get('high_risks', 0) > 0:
            recommendations.append("مخاطر عالية - ينصح بعقد اجتماع طارئ لفريق إدارة المخاطر")
        
        # توصيات بناءً على أداء الفريق
        team_analysis = self.analyze_team_productivity()
        overloaded_members = [m for m in team_analysis.get('members', {}).values()
                             if m['workload'] > 80]
        
        if overloaded_members:
            names = [m['name'] for m in overloaded_members[:3]]
            recommendations.append(f"أعضاء الفريق {', '.join(names)} لديهم عبء عمل مرتفع - ينصح بإعادة توزيع المهام")
        
        return recommendations
    
    # ============= Visualization Module =============
    def plot_project_timeline(self, project_id: str):
        """رسم الجدول الزمني للمشروع"""
        if project_id not in self.projects:
            return
        
        project_tasks = [task for task in self.tasks.values()
                        if task.project_id == project_id]
        
        fig, ax = plt.subplots(figsize=(15, len(project_tasks) * 0.5))
        
        y_pos = np.arange(len(project_tasks))
        
        for i, task in enumerate(project_tasks):
            start = task.created_at
            duration = (task.due_date - start).days
            completed = (task.completed_at - start).days if task.completed_at else 0
            
            # رسم المدة المخططة
            ax.barh(i, duration, left=start, height=0.3, 
                   color='lightgray', label='مخطط' if i == 0 else '')
            
            # رسم المدة المنجزة
            if completed > 0:
                ax.barh(i, completed, left=start, height=0.3,
                       color='green', label='منجز' if i == 0 else '')
            
            # تحديد حالة المهمة
            ax.text(start, i, f' {task.title[:30]}', va='center', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([task.id for task in project_tasks])
        ax.set_xlabel('التاريخ')
        ax.set_title(f'الجدول الزمني للمشروع: {self.projects[project_id].name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'project_timeline_{project_id}.png')
        plt.close()
    
    def plot_task_distribution(self, project_id: str):
        """رسم توزيع المهام"""
        if project_id not in self.projects:
            return
        
        project_tasks = [task for task in self.tasks.values()
                        if task.project_id == project_id]
        
        # توزيع حسب الحالة
        status_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        assignee_counts = defaultdict(int)
        
        for task in project_tasks:
            status_counts[task.status.value] += 1
            priority_counts[task.priority.value] += 1
            assignee_counts[task.assigned_to] += 1
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # رسم توزيع الحالات
        axes[0].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
        axes[0].set_title('توزيع المهام حسب الحالة')
        
        # رسم توزيع الأولويات
        axes[1].bar(priority_counts.keys(), priority_counts.values())
        axes[1].set_title('توزيع المهام حسب الأولوية')
        axes[1].tick_params(axis='x', rotation=45)
        
        # رسم توزيع المسند إليهم
        axes[2].bar(range(len(assignee_counts)), list(assignee_counts.values()))
        axes[2].set_xticks(range(len(assignee_counts)))
        axes[2].set_xticklabels([self.team_members.get(a, a).name[:10] 
                                 for a in assignee_counts.keys()], rotation=45)
        axes[2].set_title('توزيع المهام على الفريق')
        
        plt.tight_layout()
        plt.savefig(f'task_distribution_{project_id}.png')
        plt.close()
    
    def plot_burndown_chart(self, project_id: str):
        """رسم مخطط الاحتراق (Burndown Chart)"""
        if project_id not in self.projects:
            return
        
        project_tasks = [task for task in self.tasks.values()
                        if task.project_id == project_id]
        
        # حساب النقاط الزمنية
        start_date = self.projects[project_id].start_date
        end_date = self.projects[project_id].end_date or (datetime.now() + timedelta(days=30))
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # الخط المثالي
        total_hours = sum(task.estimated_hours for task in project_tasks)
        ideal_burndown = np.linspace(total_hours, 0, len(date_range))
        
        # الخط الفعلي
        actual_burndown = []
        for date in date_range:
            completed_hours = sum(task.actual_hours for task in project_tasks
                                 if task.completed_at and task.completed_at <= date)
            actual_burndown.append(total_hours - completed_hours)
        
        plt.figure(figsize=(12, 6))
        plt.plot(date_range, ideal_burndown, 'g--', label='المسار المثالي', linewidth=2)
        plt.plot(date_range, actual_burndown, 'b-', label='المسار الفعلي', linewidth=2)
        plt.fill_between(date_range, ideal_burndown, actual_burndown, 
                         where=(np.array(actual_burndown) > ideal_burndown),
                         color='red', alpha=0.3, label='تأخير')
        plt.fill_between(date_range, ideal_burndown, actual_burndown,
                         where=(np.array(actual_burndown) <= ideal_burndown),
                         color='green', alpha=0.3, label='تقدم')
        
        plt.xlabel('التاريخ')
        plt.ylabel('الساعات المتبقية')
        plt.title(f'مخطط الاحتراق - {self.projects[project_id].name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'burndown_chart_{project_id}.png')
        plt.close()
    
    def plot_team_performance_radar(self):
        """رسم رادار أداء الفريق"""
        if not self.team_members:
            return
        
        # اختيار أفضل 5 أعضاء
        top_members = sorted(self.team_members.values(), 
                            key=lambda x: x.performance_score, 
                            reverse=True)[:5]
        
        categories = ['الإنتاجية', 'الدقة', 'الالتزام', 'المهارات', 'العمل الجماعي']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for member in top_members:
            values = [
                member.performance_score,
                self.calculate_accuracy_score(member.id),
                self.calculate_commitment_score(member.id),
                len(member.skills) * 20,  # تحويل المهارات إلى نسبة
                self.calculate_collaboration_score(member.id)
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=member.name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('رادار أداء الفريق', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.savefig('team_performance_radar.png')
        plt.close()
    
    def calculate_accuracy_score(self, member_id: str) -> float:
        """حساب درجة دقة التقديرات"""
        member_tasks = [task for task in self.tasks.values()
                       if task.assigned_to == member_id and task.status == TaskStatus.DONE]
        
        if not member_tasks:
            return 70.0  # قيمة افتراضية
        
        accuracy_sum = 0
        for task in member_tasks:
            if task.estimated_hours > 0:
                accuracy = 1 - abs(task.actual_hours - task.estimated_hours) / task.estimated_hours
                accuracy_sum += max(0, accuracy) * 100
        
        return accuracy_sum / len(member_tasks)
    
    def calculate_commitment_score(self, member_id: str) -> float:
        """حساب درجة الالتزام بالمواعيد"""
        member_tasks = [task for task in self.tasks.values()
                       if task.assigned_to == member_id and task.completed_at]
        
        if not member_tasks:
            return 70.0
        
        on_time = sum(1 for task in member_tasks
                     if task.completed_at <= task.due_date)
        
        return (on_time / len(member_tasks)) * 100
    
    def calculate_collaboration_score(self, member_id: str) -> float:
        """حساب درجة التعاون مع الفريق"""
        # يمكن تحسين هذا بناءً على التفاعلات المسجلة
        member_tasks = [task for task in self.tasks.values()
                       if task.assigned_to == member_id]
        
        # عدد المهام التي تعتمد على الآخرين
        dependencies = sum(len(task.dependencies) for task in member_tasks)
        
        return min(100, 70 + dependencies * 5)
    
    # ============= Utility Functions =============
    def generate_id(self, prefix: str) -> str:
        """توليد معرف فريد"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:4]
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    def log_activity(self, action: str, details: dict):
        """تسجيل النشاطات"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        self.activity_log.append(log_entry)
    
    def save_data(self, filename: str = 'project_data.json'):
        """حفظ البيانات"""
        data = {
            'projects': {pid: p.to_dict() for pid, p in self.projects.items()},
            'tasks': {tid: t.to_dict() for tid, t in self.tasks.items()},
            'team_members': {mid: m.to_dict() for mid, m in self.team_members.items()},
            'activity_log': self.activity_log
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"تم حفظ البيانات في {filename}")
    
    def load_data(self, filename: str = 'project_data.json'):
        """تحميل البيانات"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # تحميل المشاريع
            for pid, pdata in data['projects'].items():
                project = Project(
                    id=pid,
                    name=pdata['name'],
                    description=pdata['description'],
                    manager=pdata['manager'],
                    status=ProjectStatus(pdata['status']),
                    start_date=datetime.fromisoformat(pdata['start_date']),
                    end_date=datetime.fromisoformat(pdata['end_date']) if pdata['end_date'] else None,
                    budget=pdata['budget'],
                    spent=pdata['spent'],
                    team_members=pdata['team_members'],
                    tasks=pdata['tasks'],
                    risks=pdata['risks'],
                    milestones=pdata['milestones']
                )
                self.projects[pid] = project
            
            # تحميل المهام
            for tid, tdata in data['tasks'].items():
                task = Task(
                    id=tid,
                    title=tdata['title'],
                    description=tdata['description'],
                    project_id=tdata['project_id'],
                    assigned_to=tdata['assigned_to'],
                    status=TaskStatus(tdata['status']),
                    priority=TaskPriority(tdata['priority']),
                    created_at=datetime.fromisoformat(tdata['created_at']),
                    due_date=datetime.fromisoformat(tdata['due_date']),
                    completed_at=datetime.fromisoformat(tdata['completed_at']) if tdata['completed_at'] else None,
                    estimated_hours=tdata['estimated_hours'],
                    actual_hours=tdata['actual_hours'],
                    dependencies=tdata['dependencies'],
                    tags=tdata['tags']
                )
                self.tasks[tid] = task
            
            # تحميل أعضاء الفريق
            for mid, mdata in data['team_members'].items():
                member = TeamMember(
                    id=mid,
                    name=mdata['name'],
                    email=mdata['email'],
                    role=mdata['role'],
                    skills=mdata['skills'],
                    current_projects=mdata['current_projects'],
                    workload=mdata['workload'],
                    performance_score=mdata['performance_score']
                )
                self.team_members[mid] = member
            
            self.activity_log = data['activity_log']
            self.logger.info(f"تم تحميل البيانات من {filename}")
            
        except FileNotFoundError:
            self.logger.warning(f"الملف {filename} غير موجود")

# main.py
def main():
    # إنشاء نظام إدارة المشاريع
    pms = ProjectManagementSystem()
    
    print("=== نظام إدارة المشاريع وتحليل الإنتاجية المتقدم ===\n")
    
    # 1. إضافة أعضاء الفريق
    print("1. إضافة أعضاء الفريق...")
    ahmed_id = pms.add_team_member("أحمد محمد", "ahmed@company.com", "مطور رئيسي", 
                                   ["Python", "Django", "React"])
    sara_id = pms.add_team_member("سارة علي", "sara@company.com", "محلل بيانات", 
                                  ["Python", "SQL", "Power BI"])
    khaled_id = pms.add_team_member("خالد عمر", "khaled@company.com", "مدير مشروع", 
                                    ["إدارة", "Scrum", "تحليل"])
    
    # 2. إنشاء مشروع جديد
    print("\n2. إنشاء مشروع جديد...")
    project_id = pms.create_project(
        name="تطوير منصة التجارة الإلكترونية",
        description="منصة متكاملة للتجارة الإلكترونية مع لوحة تحكم",
        manager=khaled_id,
        start_date=datetime.now(),
        budget=500000
    )
    
    # 3. إضافة أعضاء الفريق للمشروع
    print("\n3. إضافة أعضاء الفريق للمشروع...")
    pms.add_team_member_to_project(project_id, ahmed_id)
    pms.add_team_member_to_project(project_id, sara_id)
    
    # 4. إنشاء المهام
    print("\n4. إنشاء المهام...")
    task1_id = pms.create_task(
        title="تصميم قاعدة البيانات",
        description="تصميم مخطط قاعدة البيانات الرئيسية",
        project_id=project_id,
        assigned_to=ahmed_id,
        priority=TaskPriority.HIGH,
        due_date=datetime.now() + timedelta(days=7),
        estimated_hours=40
    )
    
    task2_id = pms.create_task(
        title="تطوير واجهة المستخدم",
        description="تطوير واجهات المستخدم الرئيسية",
        project_id=project_id,
        assigned_to=ahmed_id,
        priority=TaskPriority.HIGH,
        due_date=datetime.now() + timedelta(days=14),
        estimated_hours=80,
        dependencies=[task1_id]
    )
    
    task3_id = pms.create_task(
        title="تحليل البيانات",
        description="تحليل بيانات العملاء والمنتجات",
        project_id=project_id,
        assigned_to=sara_id,
        priority=TaskPriority.MEDIUM,
        due_date=datetime.now() + timedelta(days=10),
        estimated_hours=30
    )
    
    # 5. تحديث حالة المهام
    print("\n5. تحديث حالة المهام...")
    pms.update_task_status(task1_id, TaskStatus.IN_PROGRESS)
    pms.log_task_hours(task1_id, 20)
    
    pms.update_task_status(task3_id, TaskStatus.DONE)
    pms.log_task_hours(task3_id, 25)
    
    # 6. إضافة مخاطر
    print("\n6. إضافة مخاطر المشروع...")
    pms.add_risk(
        project_id=project_id,
        risk_description="تأخير في تسليم قاعدة البيانات",
        probability=0.3,
        impact=0.8,
        mitigation="تكثيف العمل على قاعدة البيانات"
    )
    
    # 7. تحليل تقدم المشروع
    print("\n7. تحليل تقدم المشروع...")
    progress = pms.analyze_project_progress(project_id)
    print(f"نسبة الإنجاز: {progress['progress']:.1f}%")
    print(f"المهام المكتملة: {progress['completed_tasks']}/{progress['total_tasks']}")
    print(f"الساعات المتبقية: {progress['remaining_hours']:.0f} ساعة")
    
    # 8. تحليل أداء الفريق
    print("\n8. تحليل أداء الفريق...")
    team_performance = pms.analyze_team_productivity()
    print(f"متوسط أداء الفريق: {team_performance['avg_performance']:.1f}%")
    
    # 9. حساب المسار الحرج
    print("\n9. حساب المسار الحرج...")
    critical_path = pms.calculate_critical_path(project_id)
    print(f"المسار الحرج: {' -> '.join(critical_path)}")
    
    # 10. التنبؤ بتاريخ الاكتمال
    print("\n10. التنبؤ بتاريخ الاكتمال...")
    predicted_date = pms.predict_completion_date(project_id)
    if predicted_date:
        print(f"تاريخ الاكتمال المتوقع: {predicted_date.strftime('%Y-%m-%d')}")
    
    # 11. توليد تقرير المشروع
    print("\n11. توليد تقرير المشروع...")
    report = pms.generate_project_report(project_id)
    
    print("\nمؤشرات الأداء الرئيسية:")
    print(f"SPI: {report['kpis']['schedule_performance_index']:.2f}")
    print(f"CPI: {report['kpis']['cost_performance_index']:.2f}")
    
    print("\nالتوصيات:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    # 12. إنشاء التصورات
    print("\n12. إنشاء التصورات البيانية...")
    pms.plot_project_timeline(project_id)
    pms.plot_task_distribution(project_id)
    pms.plot_burndown_chart(project_id)
    pms.plot_team_performance_radar()
    
    # 13. حفظ البيانات
    print("\n13. حفظ البيانات...")
    pms.save_data()
    
    print("\n✓ تم إكمال إدارة المشروع بنجاح!")
    print("تم إنشاء الملفات التالية:")
    print("- project_data.json (بيانات المشروع)")
    print("- project_timeline_PROJ_*.png (الجدول الزمني)")
    print("- task_distribution_PROJ_*.png (توزيع المهام)")
    print("- burndown_chart_PROJ_*.png (مخطط الاحتراق)")
    print("- team_performance_radar.png (رادار أداء الفريق)")
    print("- project_management.log (سجل النظام)")

if __name__ == "__main__":
    main()