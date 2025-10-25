import json
import sqlite3
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from .base_agent import BaseAgent, AgentInput, AgentOutput

class MemoryAgent(BaseAgent):
    """Agent for handling data persistence and retrieval"""
    
    def __init__(self, db_path: str = "data/memory.db"):
        super().__init__()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Memory operations
        self.operations = {
            'store': self._store_data,
            'retrieve': self._retrieve_data,
            'search': self._search_data,
            'update': self._update_data,
            'delete': self._delete_data,
            'list': self._list_data,
            'cleanup': self._cleanup_old_data,
            'export': self._export_data,
            'import': self._import_data,
            'stats': self._get_stats
        }
        
    def run(self, input: AgentInput) -> AgentOutput:
        """Execute memory operation"""
        try:
            operation = input.context.get('operation', 'store')
            
            if operation not in self.operations:
                return self._create_output(
                    success=False,
                    data={},
                    error=f"Unknown operation: {operation}",
                    confidence=0.0
                )
            
            # Execute the operation
            result = self.operations[operation](input)
            
            return self._create_output(
                success=True,
                data={
                    'operation': operation,
                    'result': result,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {
                        'db_path': str(self.db_path),
                        'operation_type': operation
                    }
                },
                confidence=0.95
            )
            
        except Exception as e:
            return self._handle_error(e, input)
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Main data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memory_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        data_type TEXT NOT NULL,
                        data_json TEXT,
                        data_blob BLOB,
                        metadata_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        tags TEXT,
                        hash TEXT,
                        size_bytes INTEGER
                    )
                ''')
                
                # Analysis history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_id TEXT,
                        analysis_type TEXT NOT NULL,
                        input_data_json TEXT,
                        output_data_json TEXT,
                        scores_json TEXT,
                        confidence REAL,
                        processing_time_ms INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata_json TEXT
                    )
                ''')
                
                # User profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT UNIQUE NOT NULL,
                        profile_data_json TEXT,
                        preferences_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_analysis_at TIMESTAMP,
                        analysis_count INTEGER DEFAULT 0
                    )
                ''')
                
                # Embeddings table for vector storage
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        embedding_blob BLOB,
                        dimension INTEGER,
                        model_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata_json TEXT
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_key ON memory_data(key)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_data(data_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_data(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_expires ON memory_data(expires_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_session ON analysis_history(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_user ON analysis_history(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_history(analysis_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_profiles_user ON user_profiles(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_key ON embeddings(key)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _store_data(self, input: AgentInput) -> Dict[str, Any]:
        """Store data in memory"""
        try:
            key = input.context.get('key')
            data = input.context.get('data')
            data_type = input.context.get('data_type', 'general')
            metadata = input.context.get('metadata', {})
            expires_in_hours = input.context.get('expires_in_hours')
            tags = input.context.get('tags', [])
            
            if not key or data is None:
                raise ValueError("Key and data are required for storage")
            
            # Calculate expiration
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.now() + timedelta(hours=expires_in_hours)
            
            # Prepare data for storage
            data_json = None
            data_blob = None
            
            try:
                # Try to store as JSON first
                data_json = json.dumps(data)
            except (TypeError, ValueError):
                # Fall back to pickle for complex objects
                data_blob = pickle.dumps(data)
            
            # Calculate hash and size
            data_str = data_json or str(data_blob)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            size_bytes = len(data_str.encode())
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO memory_data 
                    (key, data_type, data_json, data_blob, metadata_json, 
                     expires_at, tags, hash, size_bytes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    key, data_type, data_json, data_blob,
                    json.dumps(metadata), expires_at,
                    json.dumps(tags), data_hash, size_bytes
                ))
                
                conn.commit()
            
            return {
                'key': key,
                'stored': True,
                'data_type': data_type,
                'size_bytes': size_bytes,
                'hash': data_hash,
                'expires_at': expires_at.isoformat() if expires_at else None
            }
            
        except Exception as e:
            self.logger.error(f"Data storage failed: {e}")
            raise
    
    def _retrieve_data(self, input: AgentInput) -> Dict[str, Any]:
        """Retrieve data from memory"""
        try:
            key = input.context.get('key')
            include_metadata = input.context.get('include_metadata', False)
            
            if not key:
                raise ValueError("Key is required for retrieval")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT data_type, data_json, data_blob, metadata_json, 
                           created_at, updated_at, expires_at, tags, hash, size_bytes
                    FROM memory_data 
                    WHERE key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ''', (key,))
                
                row = cursor.fetchone()
                
                if not row:
                    return {
                        'key': key,
                        'found': False,
                        'data': None
                    }
                
                # Unpack row data
                data_type, data_json, data_blob, metadata_json, created_at, updated_at, expires_at, tags, data_hash, size_bytes = row
                
                # Deserialize data
                if data_json:
                    data = json.loads(data_json)
                elif data_blob:
                    data = pickle.loads(data_blob)
                else:
                    data = None
                
                result = {
                    'key': key,
                    'found': True,
                    'data': data,
                    'data_type': data_type
                }
                
                if include_metadata:
                    result['metadata'] = {
                        'created_at': created_at,
                        'updated_at': updated_at,
                        'expires_at': expires_at,
                        'tags': json.loads(tags) if tags else [],
                        'hash': data_hash,
                        'size_bytes': size_bytes,
                        'custom_metadata': json.loads(metadata_json) if metadata_json else {}
                    }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Data retrieval failed: {e}")
            raise
    
    def _search_data(self, input: AgentInput) -> Dict[str, Any]:
        """Search data in memory"""
        try:
            query = input.context.get('query', '')
            data_type = input.context.get('data_type')
            tags = input.context.get('tags', [])
            limit = input.context.get('limit', 50)
            include_data = input.context.get('include_data', False)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build search query
                where_conditions = ["(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"]
                params = []
                
                if query:
                    where_conditions.append("(key LIKE ? OR metadata_json LIKE ?)")
                    params.extend([f'%{query}%', f'%{query}%'])
                
                if data_type:
                    where_conditions.append("data_type = ?")
                    params.append(data_type)
                
                if tags:
                    for tag in tags:
                        where_conditions.append("tags LIKE ?")
                        params.append(f'%{tag}%')
                
                where_clause = " AND ".join(where_conditions)
                
                if include_data:
                    select_fields = "key, data_type, data_json, data_blob, metadata_json, created_at, updated_at, tags"
                else:
                    select_fields = "key, data_type, metadata_json, created_at, updated_at, tags, size_bytes"
                
                cursor.execute(f'''
                    SELECT {select_fields}
                    FROM memory_data 
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT ?
                ''', params + [limit])
                
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    if include_data:
                        key, data_type, data_json, data_blob, metadata_json, created_at, updated_at, tags = row
                        
                        # Deserialize data
                        if data_json:
                            data = json.loads(data_json)
                        elif data_blob:
                            data = pickle.loads(data_blob)
                        else:
                            data = None
                        
                        result_item = {
                            'key': key,
                            'data_type': data_type,
                            'data': data,
                            'metadata': json.loads(metadata_json) if metadata_json else {},
                            'created_at': created_at,
                            'updated_at': updated_at,
                            'tags': json.loads(tags) if tags else []
                        }
                    else:
                        key, data_type, metadata_json, created_at, updated_at, tags, size_bytes = row
                        
                        result_item = {
                            'key': key,
                            'data_type': data_type,
                            'metadata': json.loads(metadata_json) if metadata_json else {},
                            'created_at': created_at,
                            'updated_at': updated_at,
                            'tags': json.loads(tags) if tags else [],
                            'size_bytes': size_bytes
                        }
                    
                    results.append(result_item)
                
                return {
                    'query': query,
                    'results': results,
                    'count': len(results),
                    'limit': limit
                }
                
        except Exception as e:
            self.logger.error(f"Data search failed: {e}")
            raise
    
    def _update_data(self, input: AgentInput) -> Dict[str, Any]:
        """Update existing data in memory"""
        try:
            key = input.context.get('key')
            data = input.context.get('data')
            metadata = input.context.get('metadata')
            tags = input.context.get('tags')
            
            if not key:
                raise ValueError("Key is required for update")
            
            # Check if key exists
            existing = self._retrieve_data(AgentInput(context={'key': key}))
            if not existing['found']:
                raise ValueError(f"Key '{key}' not found")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query
                update_fields = ["updated_at = CURRENT_TIMESTAMP"]
                params = []
                
                if data is not None:
                    # Prepare data for storage
                    try:
                        data_json = json.dumps(data)
                        update_fields.extend(["data_json = ?", "data_blob = NULL"])
                        params.append(data_json)
                    except (TypeError, ValueError):
                        data_blob = pickle.dumps(data)
                        update_fields.extend(["data_blob = ?", "data_json = NULL"])
                        params.append(data_blob)
                    
                    # Update hash and size
                    data_str = data_json if 'data_json' in locals() else str(data_blob)
                    data_hash = hashlib.sha256(data_str.encode()).hexdigest()
                    size_bytes = len(data_str.encode())
                    
                    update_fields.extend(["hash = ?", "size_bytes = ?"])
                    params.extend([data_hash, size_bytes])
                
                if metadata is not None:
                    update_fields.append("metadata_json = ?")
                    params.append(json.dumps(metadata))
                
                if tags is not None:
                    update_fields.append("tags = ?")
                    params.append(json.dumps(tags))
                
                params.append(key)
                
                cursor.execute(f'''
                    UPDATE memory_data 
                    SET {', '.join(update_fields)}
                    WHERE key = ?
                ''', params)
                
                conn.commit()
                
                return {
                    'key': key,
                    'updated': True,
                    'changes_made': len(update_fields) - 1  # Exclude updated_at
                }
                
        except Exception as e:
            self.logger.error(f"Data update failed: {e}")
            raise
    
    def _delete_data(self, input: AgentInput) -> Dict[str, Any]:
        """Delete data from memory"""
        try:
            key = input.context.get('key')
            keys = input.context.get('keys', [])
            
            if key:
                keys = [key]
            elif not keys:
                raise ValueError("Key or keys are required for deletion")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                deleted_count = 0
                for k in keys:
                    cursor.execute("DELETE FROM memory_data WHERE key = ?", (k,))
                    if cursor.rowcount > 0:
                        deleted_count += 1
                
                conn.commit()
                
                return {
                    'keys': keys,
                    'deleted_count': deleted_count,
                    'total_requested': len(keys)
                }
                
        except Exception as e:
            self.logger.error(f"Data deletion failed: {e}")
            raise
    
    def _list_data(self, input: AgentInput) -> Dict[str, Any]:
        """List data in memory"""
        try:
            data_type = input.context.get('data_type')
            limit = input.context.get('limit', 100)
            offset = input.context.get('offset', 0)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                where_clause = "(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"
                params = []
                
                if data_type:
                    where_clause += " AND data_type = ?"
                    params.append(data_type)
                
                cursor.execute(f'''
                    SELECT key, data_type, created_at, updated_at, 
                           expires_at, tags, size_bytes
                    FROM memory_data 
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                ''', params + [limit, offset])
                
                rows = cursor.fetchall()
                
                # Get total count
                cursor.execute(f'''
                    SELECT COUNT(*) FROM memory_data WHERE {where_clause}
                ''', params)
                
                total_count = cursor.fetchone()[0]
                
                items = []
                for row in rows:
                    key, data_type, created_at, updated_at, expires_at, tags, size_bytes = row
                    
                    items.append({
                        'key': key,
                        'data_type': data_type,
                        'created_at': created_at,
                        'updated_at': updated_at,
                        'expires_at': expires_at,
                        'tags': json.loads(tags) if tags else [],
                        'size_bytes': size_bytes
                    })
                
                return {
                    'items': items,
                    'count': len(items),
                    'total_count': total_count,
                    'limit': limit,
                    'offset': offset,
                    'has_more': offset + len(items) < total_count
                }
                
        except Exception as e:
            self.logger.error(f"Data listing failed: {e}")
            raise
    
    def _cleanup_old_data(self, input: AgentInput) -> Dict[str, Any]:
        """Clean up expired and old data"""
        try:
            max_age_days = input.context.get('max_age_days', 30)
            dry_run = input.context.get('dry_run', False)
            
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find expired and old data
                cursor.execute('''
                    SELECT key, data_type, created_at, expires_at, size_bytes
                    FROM memory_data 
                    WHERE expires_at < CURRENT_TIMESTAMP 
                       OR created_at < ?
                    ORDER BY created_at
                ''', (cutoff_date,))
                
                rows = cursor.fetchall()
                
                if dry_run:
                    return {
                        'dry_run': True,
                        'items_to_delete': len(rows),
                        'total_size_bytes': sum(row[4] for row in rows),
                        'items': [{
                            'key': row[0],
                            'data_type': row[1],
                            'created_at': row[2],
                            'expires_at': row[3],
                            'size_bytes': row[4]
                        } for row in rows]
                    }
                
                # Delete the data
                deleted_count = 0
                total_size_freed = 0
                
                for row in rows:
                    cursor.execute("DELETE FROM memory_data WHERE key = ?", (row[0],))
                    if cursor.rowcount > 0:
                        deleted_count += 1
                        total_size_freed += row[4]
                
                conn.commit()
                
                return {
                    'deleted_count': deleted_count,
                    'total_size_freed_bytes': total_size_freed,
                    'max_age_days': max_age_days
                }
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
            raise
    
    def _export_data(self, input: AgentInput) -> Dict[str, Any]:
        """Export data from memory"""
        try:
            export_path = input.context.get('export_path', 'memory_export.json')
            data_type = input.context.get('data_type')
            include_expired = input.context.get('include_expired', False)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                where_conditions = []
                params = []
                
                if not include_expired:
                    where_conditions.append("(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)")
                
                if data_type:
                    where_conditions.append("data_type = ?")
                    params.append(data_type)
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                cursor.execute(f'''
                    SELECT key, data_type, data_json, data_blob, metadata_json,
                           created_at, updated_at, expires_at, tags
                    FROM memory_data 
                    WHERE {where_clause}
                    ORDER BY created_at
                ''', params)
                
                rows = cursor.fetchall()
                
                # Prepare export data
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_items': len(rows),
                    'data_type_filter': data_type,
                    'include_expired': include_expired,
                    'items': []
                }
                
                for row in rows:
                    key, data_type, data_json, data_blob, metadata_json, created_at, updated_at, expires_at, tags = row
                    
                    # Deserialize data
                    if data_json:
                        data = json.loads(data_json)
                    elif data_blob:
                        # Convert blob to base64 for JSON serialization
                        import base64
                        data = {
                            '_type': 'blob',
                            '_data': base64.b64encode(data_blob).decode('utf-8')
                        }
                    else:
                        data = None
                    
                    export_data['items'].append({
                        'key': key,
                        'data_type': data_type,
                        'data': data,
                        'metadata': json.loads(metadata_json) if metadata_json else {},
                        'created_at': created_at,
                        'updated_at': updated_at,
                        'expires_at': expires_at,
                        'tags': json.loads(tags) if tags else []
                    })
                
                # Write to file
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                return {
                    'export_path': export_path,
                    'exported_items': len(rows),
                    'file_size_bytes': Path(export_path).stat().st_size
                }
                
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            raise
    
    def _import_data(self, input: AgentInput) -> Dict[str, Any]:
        """Import data into memory"""
        try:
            import_path = input.context.get('import_path')
            overwrite_existing = input.context.get('overwrite_existing', False)
            
            if not import_path or not Path(import_path).exists():
                raise ValueError("Valid import_path is required")
            
            # Load import data
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if 'items' not in import_data:
                raise ValueError("Invalid import file format")
            
            imported_count = 0
            skipped_count = 0
            error_count = 0
            
            for item in import_data['items']:
                try:
                    key = item['key']
                    
                    # Check if key exists
                    if not overwrite_existing:
                        existing = self._retrieve_data(AgentInput(context={'key': key}))
                        if existing['found']:
                            skipped_count += 1
                            continue
                    
                    # Prepare data
                    data = item['data']
                    if isinstance(data, dict) and data.get('_type') == 'blob':
                        # Convert base64 back to blob
                        import base64
                        data = base64.b64decode(data['_data'])
                    
                    # Store the item
                    store_input = AgentInput(context={
                        'key': key,
                        'data': data,
                        'data_type': item['data_type'],
                        'metadata': item.get('metadata', {}),
                        'tags': item.get('tags', [])
                    })
                    
                    self._store_data(store_input)
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to import item {item.get('key', 'unknown')}: {e}")
                    error_count += 1
            
            return {
                'import_path': import_path,
                'total_items': len(import_data['items']),
                'imported_count': imported_count,
                'skipped_count': skipped_count,
                'error_count': error_count
            }
            
        except Exception as e:
            self.logger.error(f"Data import failed: {e}")
            raise
    
    def _get_stats(self, input: AgentInput) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM memory_data")
                total_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memory_data WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP")
                active_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memory_data WHERE expires_at < CURRENT_TIMESTAMP")
                expired_items = cursor.fetchone()[0]
                
                # Size statistics
                cursor.execute("SELECT SUM(size_bytes), AVG(size_bytes) FROM memory_data")
                size_row = cursor.fetchone()
                total_size = size_row[0] or 0
                avg_size = size_row[1] or 0
                
                # Data type breakdown
                cursor.execute("SELECT data_type, COUNT(*) FROM memory_data GROUP BY data_type")
                data_types = dict(cursor.fetchall())
                
                # Recent activity
                cursor.execute("SELECT COUNT(*) FROM memory_data WHERE created_at > datetime('now', '-24 hours')")
                items_last_24h = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memory_data WHERE updated_at > datetime('now', '-24 hours')")
                updates_last_24h = cursor.fetchone()[0]
                
                # Analysis history stats
                cursor.execute("SELECT COUNT(*) FROM analysis_history")
                total_analyses = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM analysis_history WHERE user_id IS NOT NULL")
                unique_users = cursor.fetchone()[0]
                
                # User profiles stats
                cursor.execute("SELECT COUNT(*) FROM user_profiles")
                total_profiles = cursor.fetchone()[0]
                
                # Embeddings stats
                cursor.execute("SELECT COUNT(*) FROM embeddings")
                total_embeddings = cursor.fetchone()[0]
                
                return {
                    'memory_data': {
                        'total_items': total_items,
                        'active_items': active_items,
                        'expired_items': expired_items,
                        'total_size_bytes': total_size,
                        'average_size_bytes': avg_size,
                        'data_types': data_types,
                        'items_created_last_24h': items_last_24h,
                        'items_updated_last_24h': updates_last_24h
                    },
                    'analysis_history': {
                        'total_analyses': total_analyses,
                        'unique_users': unique_users
                    },
                    'user_profiles': {
                        'total_profiles': total_profiles
                    },
                    'embeddings': {
                        'total_embeddings': total_embeddings
                    },
                    'database': {
                        'path': str(self.db_path),
                        'size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Stats generation failed: {e}")
            raise
    
    # Additional helper methods for specific use cases
    
    def store_analysis_result(self, session_id: str, user_id: Optional[str], 
                            analysis_type: str, input_data: Dict[str, Any], 
                            output_data: Dict[str, Any], scores: Dict[str, float],
                            confidence: float, processing_time_ms: int,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store analysis result in history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO analysis_history 
                    (session_id, user_id, analysis_type, input_data_json, 
                     output_data_json, scores_json, confidence, processing_time_ms, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, user_id, analysis_type,
                    json.dumps(input_data), json.dumps(output_data),
                    json.dumps(scores), confidence, processing_time_ms,
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Analysis result storage failed: {e}")
            return False
    
    def store_user_profile(self, user_id: str, profile_data: Dict[str, Any],
                          preferences: Optional[Dict[str, Any]] = None) -> bool:
        """Store or update user profile"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_profiles 
                    (user_id, profile_data_json, preferences_json, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id, json.dumps(profile_data),
                    json.dumps(preferences or {})
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"User profile storage failed: {e}")
            return False
    
    def store_embedding(self, key: str, embedding: List[float], 
                       model_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store vector embedding"""
        try:
            import numpy as np
            
            embedding_blob = pickle.dumps(np.array(embedding))
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO embeddings 
                    (key, embedding_blob, dimension, model_name, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    key, embedding_blob, len(embedding), model_name,
                    json.dumps(metadata or {})
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Embedding storage failed: {e}")
            return False
    
    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Retrieve vector embedding"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT embedding_blob FROM embeddings WHERE key = ?", (key,))
                row = cursor.fetchone()
                
                if row:
                    import numpy as np
                    embedding_array = pickle.loads(row[0])
                    return embedding_array.tolist()
                
                return None
                
        except Exception as e:
            self.logger.error(f"Embedding retrieval failed: {e}")
            return None
