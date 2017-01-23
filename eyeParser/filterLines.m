function lines = filterLines(lines_content, regExp)
% Filters lines of a file content by a regular expression
% Check if regex is present
    chechPresent = @(x) any(x==1);
    lines = find(cellfun(chechPresent, regexp(lines_content, regExp)));
end