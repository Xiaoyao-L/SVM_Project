function [temp_labels] = change_label(Ytr,classname)
temp_labels = Ytr;
temp2 = find(Ytr ~= classname);
temp1 = find(Ytr == classname);

temp_labels(temp2) = -1;
temp_labels(temp1) = 1;
end

