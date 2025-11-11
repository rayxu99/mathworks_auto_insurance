function projRoot = getProjRoot()
%GETPROJROOT Return the folder that contains the running .mlx or .m file.
%
%   projRoot = GETPROJROOT() looks at the active document in the MATLAB
%   Editor.  If the code is being executed from the temporary "Editor_…"
%   copy, it maps back to the *original* file that you opened.  The
%   function throws an error if the active document has never been saved.

    doc = matlab.desktop.editor.getActive;

    if isempty(doc.Filename)
        error('Save the Live Script before running.');
    end

    % When a section is run, doc.Filename is the temp copy; doc.Opened is
    % the real path.  For "Run" or "Run All", they are the same.
    if contains(doc.Filename, ['T' filesep 'Editor_'])
        fullPath = doc.Opened;     % original file you opened
    else
        fullPath = doc.Filename;   % normal case
    end

    projRoot = fileparts(fullPath);   % folder that holds the live script
    fprintf('Data directory resolved to:\n%s\n', projRoot);
end
