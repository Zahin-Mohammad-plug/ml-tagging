import React, { useEffect, useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Alert,
  Divider,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Grid,
  Card,
  CardContent,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  CircularProgress,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Delete,
  Add,
  Edit,
  Search,
  Block,
  CheckCircle,
  Cancel,
  Save,
  Close,
  Settings,
  Sync,
  Download,
  CloudDownload,
  CloudUpload,
} from '@mui/icons-material';
import { toast } from 'react-hot-toast';
import axios from 'axios';

interface Tag {
  tag_id: string;
  name: string;
  description?: string;
  prompts: string[];
  aliases?: string[];
  parent_tag_ids?: string[];
  child_tag_ids?: string[];
  review_threshold?: number;
  auto_threshold?: number;
  is_active: boolean;
  is_blacklisted: boolean;
  blacklist_reason?: string;
  blacklist_id?: string;
  created_at?: string;
  updated_at?: string;
}


const Tags: React.FC = () => {
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:9898';
  
  const [tags, setTags] = useState<Tag[]>([]);
  const [filteredTags, setFilteredTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterActive, setFilterActive] = useState<string>('all'); // all, active, inactive, blacklisted
  const [sortBy, setSortBy] = useState<string>('name');
  const [sortOrder, setSortOrder] = useState<string>('asc');
  const [selectedTag, setSelectedTag] = useState<Tag | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingTag, setEditingTag] = useState<Tag | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [searchParams] = useSearchParams();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [importing, setImporting] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchTags();
  }, []);

  // Handle search parameter from URL after tags are loaded
  useEffect(() => {
    const searchParam = searchParams.get('search');
    if (searchParam && tags.length > 0 && !editDialogOpen) {
      setSearchQuery(searchParam);
      // Try to find and open the tag
      const tag = tags.find(t => t.name.toLowerCase() === searchParam.toLowerCase());
      if (tag) {
        setEditingTag({ ...tag });
        setEditDialogOpen(true);
        setTabValue(0);
      }
    }
  }, [tags, searchParams, editDialogOpen]);

  useEffect(() => {
    filterAndSortTags();
  }, [tags, searchQuery, filterActive, sortBy, sortOrder]);

  const fetchTags = async () => {
    try {
      setLoading(true);
      setError(null);
      // Don't filter by is_active in API call - we'll filter client-side to support blacklist filter
      const response = await axios.get(`${apiUrl}/tags`, {
        params: {
          limit: 1000,
        }
      });
      setTags(response.data);
    } catch (err: any) {
      console.error('Failed to fetch tags:', err);
      setError('Failed to load tags. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const filterAndSortTags = () => {
    let filtered = [...tags];

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(tag =>
        tag.name.toLowerCase().includes(query) ||
        (tag.description && tag.description.toLowerCase().includes(query))
      );
    }

    // Apply status filter
    if (filterActive === 'active') {
      filtered = filtered.filter(tag => tag.is_active);
    } else if (filterActive === 'inactive') {
      filtered = filtered.filter(tag => !tag.is_active);
    } else if (filterActive === 'blacklisted') {
      filtered = filtered.filter(tag => tag.is_blacklisted);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aVal: any, bVal: any;
      if (sortBy === 'name') {
        aVal = a.name.toLowerCase();
        bVal = b.name.toLowerCase();
      } else if (sortBy === 'created_at') {
        aVal = new Date(a.created_at || 0).getTime();
        bVal = new Date(b.created_at || 0).getTime();
      } else if (sortBy === 'updated_at') {
        aVal = new Date(a.updated_at || 0).getTime();
        bVal = new Date(b.updated_at || 0).getTime();
      } else {
        aVal = a.name.toLowerCase();
        bVal = b.name.toLowerCase();
      }

      if (sortOrder === 'desc') {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      } else {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      }
    });

    setFilteredTags(filtered);
  };

  const handleEditTag = useCallback((tag: Tag) => {
    setEditingTag({ ...tag });
    setEditDialogOpen(true);
    setTabValue(0);
  }, []);

  const handleSaveTag = async () => {
    if (!editingTag) return;

    try {
      setError(null);
      
      // Save tag first
      await axios.put(`${apiUrl}/tags/${editingTag.tag_id}`, {
        name: editingTag.name,
        description: editingTag.description,
        prompts: editingTag.prompts,
        aliases: editingTag.aliases || [],
        parent_tag_ids: editingTag.parent_tag_ids || [],
        child_tag_ids: editingTag.child_tag_ids || [],
        review_threshold: editingTag.review_threshold,
        auto_threshold: editingTag.auto_threshold,
        is_active: editingTag.is_active,
      });

      // Get the original tag to check blacklist status before edit
      const originalTag = tags.find(t => t.tag_id === editingTag.tag_id);
      const wasBlacklisted = originalTag?.is_blacklisted || false;
      const wasBlacklistReason = originalTag?.blacklist_reason || '';

      // Update blacklist status if changed
      if (editingTag.is_blacklisted && !wasBlacklisted) {
        // Add to blacklist
        await axios.post(`${apiUrl}/blacklist`, {
          tag_name: editingTag.name,
          tag_id: editingTag.tag_id,
          reason: editingTag.blacklist_reason || '',
        });
      } else if (!editingTag.is_blacklisted && wasBlacklisted) {
        // Remove from blacklist
        await axios.delete(`${apiUrl}/blacklist/${editingTag.name}`);
      } else if (editingTag.is_blacklisted && wasBlacklisted && editingTag.blacklist_reason !== wasBlacklistReason) {
        // Update blacklist reason - remove and re-add with new reason
        await axios.delete(`${apiUrl}/blacklist/${editingTag.name}`);
        await axios.post(`${apiUrl}/blacklist`, {
          tag_name: editingTag.name,
          tag_id: editingTag.tag_id,
          reason: editingTag.blacklist_reason || '',
        });
      }

      setEditDialogOpen(false);
      setEditingTag(null);
      fetchTags();
    } catch (err: any) {
      console.error('Failed to save tag:', err);
      setError(err.response?.data?.detail || 'Failed to save tag');
    }
  };

  const handleAddPrompt = () => {
    if (!editingTag) return;
    setEditingTag({
      ...editingTag,
      prompts: [...(editingTag.prompts || []), ''],
    });
  };

  const handleUpdatePrompt = (index: number, value: string) => {
    if (!editingTag) return;
    const newPrompts = [...editingTag.prompts];
    newPrompts[index] = value;
    setEditingTag({
      ...editingTag,
      prompts: newPrompts,
    });
  };

  const handleDeletePrompt = (index: number) => {
    if (!editingTag) return;
    const newPrompts = editingTag.prompts.filter((_, i) => i !== index);
    setEditingTag({
      ...editingTag,
      prompts: newPrompts,
    });
  };

  const handleExportTags = async (includePrompts: boolean) => {
    try {
      setExporting(true);
      setError(null);
      
      const response = await axios.get(`${apiUrl}/tags/export`, {
        params: {
          include_prompts: includePrompts,
        },
      });
      
      if (response.data.success) {
        // Create and download the JSON file
        const dataStr = JSON.stringify(response.data.tags, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = includePrompts 
          ? `tag_prompts_${new Date().toISOString().split('T')[0]}.json`
          : `tags_metadata_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        toast.success(
          `Exported ${response.data.count} ${includePrompts ? 'tags with prompts' : 'tags'}`,
          { duration: 3000 }
        );
      } else {
        throw new Error('Export failed');
      }
    } catch (err: any) {
      console.error('Failed to export tags:', err);
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to export tags';
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setExporting(false);
    }
  };

  const handleImportTags = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      setImporting(true);
      setError(null);

      // Validate file type
      if (!file.name.endsWith('.json')) {
        throw new Error('File must be a JSON file');
      }

      // Create FormData and upload file
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${apiUrl}/tags/import`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        const { tags_created, tags_updated, tags_skipped, errors_count, total_in_file } = response.data;
        
        let message = `Import completed: ${tags_created} created, ${tags_updated} updated`;
        if (tags_skipped > 0) {
          message += `, ${tags_skipped} skipped`;
        }
        if (errors_count > 0) {
          message += `, ${errors_count} errors`;
        }
        message += ` (${total_in_file} total in file)`;
        
        toast.success(message, { duration: 5000 });
        
        // Show errors if any
        if (response.data.errors && response.data.errors.length > 0) {
          console.warn('Import errors:', response.data.errors);
          if (response.data.errors.length <= 5) {
            toast.error(`Errors: ${response.data.errors.join('; ')}`, { duration: 8000 });
          } else {
            toast.error(`First 5 errors: ${response.data.errors.slice(0, 5).join('; ')}`, { duration: 8000 });
          }
        }
        
        // Refresh tags list
        await fetchTags();
      } else {
        throw new Error(response.data.error || 'Import failed');
      }
    } catch (err: any) {
      console.error('Failed to import tags:', err);
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to import tags';
      setError(errorMsg);
      toast.error(errorMsg);
    } finally {
      setImporting(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Tags Management
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Manage tags, prompts, blacklist, and thresholds. Click on a tag to edit.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Search and Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              placeholder="Search tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Filter</InputLabel>
              <Select
                value={filterActive}
                label="Filter"
                onChange={(e) => setFilterActive(e.target.value)}
              >
                <MenuItem value="all">All Tags</MenuItem>
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="inactive">Inactive</MenuItem>
                <MenuItem value="blacklisted">Blacklisted</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Sort By</InputLabel>
              <Select
                value={sortBy}
                label="Sort By"
                onChange={(e) => setSortBy(e.target.value)}
              >
                <MenuItem value="name">Name</MenuItem>
                <MenuItem value="created_at">Created Date</MenuItem>
                <MenuItem value="updated_at">Updated Date</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              fullWidth
              variant="outlined"
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            >
              {sortOrder === 'asc' ? '↑' : '↓'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Settings Box */}
      <Paper sx={{ p: 2, mb: 3, bgcolor: 'background.default' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Settings sx={{ mr: 1 }} />
          <Typography variant="h6">Tag Settings & Actions</Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Button
            size="small"
            onClick={() => setSettingsOpen(!settingsOpen)}
          >
            {settingsOpen ? 'Hide' : 'Show'}
          </Button>
        </Box>
        
        {settingsOpen && (
          <Box>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  style={{ display: 'none' }}
                  onChange={handleImportTags}
                />
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={importing ? <CircularProgress size={20} /> : <CloudUpload />}
                  onClick={() => fileInputRef.current?.click()}
                  disabled={importing}
                  sx={{ py: 1.5 }}
                >
                  Import Tags + Prompts
                </Button>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                  Import tags with prompts from JSON file
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={exporting ? <CircularProgress size={20} /> : <CloudDownload />}
                  onClick={() => handleExportTags(true)}
                  disabled={exporting}
                  sx={{ py: 1.5 }}
                >
                  Export Tags + Prompts
                </Button>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                  Download tags with prompts in tag_prompts.json format
                </Typography>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={exporting ? <CircularProgress size={20} /> : <Download />}
                  onClick={() => handleExportTags(false)}
                  disabled={exporting}
                  sx={{ py: 1.5 }}
                >
                  Export Tags (Metadata)
                </Button>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                  Download tag metadata without prompts
                </Typography>
              </Grid>
            </Grid>
          </Box>
        )}
      </Paper>

      {/* Tags List */}
      <Paper>
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">
            Tags ({filteredTags.length})
          </Typography>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => {
              // TODO: Add create tag functionality
              alert('Create tag functionality not yet implemented');
            }}
          >
            Add Tag
          </Button>
        </Box>
        <Divider />
        {filteredTags.length === 0 ? (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography color="text.secondary">No tags found</Typography>
          </Box>
        ) : (
          <List>
            {filteredTags.map((tag, index) => (
              <React.Fragment key={tag.tag_id}>
                <ListItem
                  button
                  onClick={() => handleEditTag(tag)}
                  sx={{
                    '&:hover': {
                      bgcolor: 'action.hover',
                    },
                  }}
                >
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle1">{tag.name}</Typography>
                        {tag.is_blacklisted && (
                          <Chip
                            icon={<Block />}
                            label="Blacklisted"
                            color="error"
                            size="small"
                          />
                        )}
                        {!tag.is_active && (
                          <Chip label="Inactive" color="default" size="small" />
                        )}
                        <Chip
                          label={`${tag.prompts?.length || 0} prompts`}
                          size="small"
                          variant="outlined"
                        />
                        {(tag.aliases && tag.aliases.length > 0) && (
                          <Chip
                            label={`${tag.aliases.length} aliases`}
                            size="small"
                            variant="outlined"
                            color="info"
                          />
                        )}
                        {(tag.parent_tag_ids && tag.parent_tag_ids.length > 0) && (
                          <Chip
                            label={`${tag.parent_tag_ids.length} parents`}
                            size="small"
                            variant="outlined"
                            color="secondary"
                          />
                        )}
                        {(tag.child_tag_ids && tag.child_tag_ids.length > 0) && (
                          <Chip
                            label={`${tag.child_tag_ids.length} children`}
                            size="small"
                            variant="outlined"
                            color="primary"
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box>
                        {tag.description && (
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                            {tag.description}
                          </Typography>
                        )}
                        <Box sx={{ display: 'flex', gap: 2, mt: 0.5, flexWrap: 'wrap' }}>
                          {tag.review_threshold !== undefined && tag.review_threshold !== null && (
                            <Typography variant="caption">
                              Review: {(tag.review_threshold * 100).toFixed(0)}%
                            </Typography>
                          )}
                          {tag.auto_threshold !== undefined && tag.auto_threshold !== null && (
                            <Typography variant="caption">
                              Auto: {(tag.auto_threshold * 100).toFixed(0)}%
                            </Typography>
                          )}
                          {tag.aliases && tag.aliases.length > 0 && (
                            <Typography variant="caption" color="text.secondary">
                              Aliases: {tag.aliases.slice(0, 3).join(', ')}{tag.aliases.length > 3 ? '...' : ''}
                            </Typography>
                          )}
                        </Box>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" onClick={(e) => {
                      e.stopPropagation();
                      handleEditTag(tag);
                    }}>
                      <Edit />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                {index < filteredTags.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        )}
      </Paper>

      {/* Edit Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={() => {
          setEditDialogOpen(false);
          setEditingTag(null);
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Edit Tag: {editingTag?.name}
        </DialogTitle>
        <DialogContent>
          {editingTag && (
            <Box sx={{ mt: 2 }}>
              <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)} sx={{ mb: 2 }}>
                <Tab label="Basic Info" />
                <Tab label="Relationships" />
                <Tab label="Prompts" />
                <Tab label="Settings" />
              </Tabs>

              {tabValue === 0 && (
                <Box>
                  <TextField
                    fullWidth
                    label="Tag Name"
                    value={editingTag.name}
                    onChange={(e) => setEditingTag({ ...editingTag, name: e.target.value })}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    label="Description"
                    value={editingTag.description || ''}
                    onChange={(e) => setEditingTag({ ...editingTag, description: e.target.value })}
                    multiline
                    rows={3}
                    sx={{ mb: 2 }}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={editingTag.is_active}
                        onChange={(e) => setEditingTag({ ...editingTag, is_active: e.target.checked })}
                      />
                    }
                    label="Active"
                  />
                </Box>
              )}

              {tabValue === 1 && (
                <Box>
                  <Typography variant="h6" sx={{ mb: 2 }}>Tag Relationships</Typography>
                  
                  <TextField
                    fullWidth
                    label="Aliases"
                    placeholder="Comma-separated aliases (e.g., alias1, alias2)"
                    value={(editingTag.aliases || []).join(', ')}
                    onChange={(e) => {
                      const aliases = e.target.value.split(',').map(a => a.trim()).filter(a => a);
                      setEditingTag({ ...editingTag, aliases });
                    }}
                    helperText="Alternative names for this tag (one per line or comma-separated)"
                    multiline
                    rows={2}
                    sx={{ mb: 3 }}
                  />
                  
                  <TextField
                    fullWidth
                    label="Parent Tag IDs"
                    placeholder="Comma-separated tag IDs (e.g., tag-id-1, tag-id-2)"
                    value={(editingTag.parent_tag_ids || []).join(', ')}
                    onChange={(e) => {
                      const parentIds = e.target.value.split(',').map(id => id.trim()).filter(id => id);
                      setEditingTag({ ...editingTag, parent_tag_ids: parentIds });
                    }}
                    helperText="Parent tags (more general categories). Enter tag IDs separated by commas."
                    multiline
                    rows={2}
                    sx={{ mb: 3 }}
                  />
                  
                  <TextField
                    fullWidth
                    label="Child Tag IDs"
                    placeholder="Comma-separated tag IDs (e.g., tag-id-1, tag-id-2)"
                    value={(editingTag.child_tag_ids || []).join(', ')}
                    onChange={(e) => {
                      const childIds = e.target.value.split(',').map(id => id.trim()).filter(id => id);
                      setEditingTag({ ...editingTag, child_tag_ids: childIds });
                    }}
                    helperText="Child tags (more specific subcategories). Enter tag IDs separated by commas."
                    multiline
                    rows={2}
                    sx={{ mb: 2 }}
                  />
                </Box>
              )}

              {tabValue === 2 && (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">Sentence Prompts</Typography>
                    <Button
                      variant="outlined"
                      startIcon={<Add />}
                      onClick={handleAddPrompt}
                    >
                      Add Prompt
                    </Button>
                  </Box>
                  {editingTag.prompts && editingTag.prompts.length > 0 ? (
                    editingTag.prompts.map((prompt, index) => (
                      <Box key={index} sx={{ mb: 2, display: 'flex', gap: 1 }}>
                        <TextField
                          fullWidth
                          label={`Prompt ${index + 1}`}
                          value={prompt}
                          onChange={(e) => handleUpdatePrompt(index, e.target.value)}
                          multiline
                          rows={2}
                        />
                        <IconButton
                          color="error"
                          onClick={() => handleDeletePrompt(index)}
                          sx={{ mt: 1 }}
                        >
                          <Delete />
                        </IconButton>
                      </Box>
                    ))
                  ) : (
                    <Alert severity="info">No prompts yet. Add prompts to help the ML model identify this tag.</Alert>
                  )}
                </Box>
              )}

              {tabValue === 2 && (
                <Box>
                  <TextField
                    fullWidth
                    type="number"
                    label="Review Threshold"
                    value={editingTag.review_threshold ?? ''}
                    onChange={(e) => setEditingTag({ ...editingTag, review_threshold: parseFloat(e.target.value) || undefined })}
                    inputProps={{ min: 0, max: 1, step: 0.1 }}
                    helperText="Minimum confidence (0.0-1.0) for this tag to appear in review"
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    type="number"
                    label="Auto Threshold"
                    value={editingTag.auto_threshold ?? ''}
                    onChange={(e) => setEditingTag({ ...editingTag, auto_threshold: parseFloat(e.target.value) || undefined })}
                    inputProps={{ min: 0, max: 1, step: 0.1 }}
                    helperText="Minimum confidence (0.0-1.0) for this tag to be auto-approved"
                    sx={{ mb: 2 }}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={editingTag.is_blacklisted}
                        onChange={(e) => setEditingTag({ ...editingTag, is_blacklisted: e.target.checked })}
                      />
                    }
                    label="Blacklisted"
                    sx={{ mb: 2, display: 'block' }}
                  />
                  {editingTag.is_blacklisted && (
                    <TextField
                      fullWidth
                      label="Blacklist Reason"
                      value={editingTag.blacklist_reason || ''}
                      onChange={(e) => setEditingTag({ ...editingTag, blacklist_reason: e.target.value })}
                      multiline
                      rows={2}
                      helperText="Reason for blacklisting this tag"
                    />
                  )}
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Tag ID: {editingTag.tag_id}
                    </Typography>
                    {editingTag.created_at && (
                      <Typography variant="caption" color="text.secondary" display="block">
                        Created: {new Date(editingTag.created_at).toLocaleString()}
                      </Typography>
                    )}
                    {editingTag.updated_at && (
                      <Typography variant="caption" color="text.secondary" display="block">
                        Updated: {new Date(editingTag.updated_at).toLocaleString()}
                      </Typography>
                    )}
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setEditDialogOpen(false);
            setEditingTag(null);
          }}>
            Cancel
          </Button>
          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={handleSaveTag}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default Tags;

