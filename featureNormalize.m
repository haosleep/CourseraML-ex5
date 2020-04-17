function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% �o��O�Q�ΤFbsxfun�Ӷi��S�x�Y��
% �|��]�j��v�C�B���٭n�ӱo���Ĳv
% bsxfun���y�k�Obsxfun(@fun, A, B)
% �ҨϥΪ�fun�����O�G�i����(�u�঳��ӿ�J)
% AB�x�}���ץ����n��۹���
% (�榡�ۦP(���Oaxb),�άO���P���榡�䤤�@���O1(���O�@��axb�@��ax1�o��))

% ��J��X�O��polyFeatures.m�M�g�᪺X_poly(�榡�O12x9)
% ���C�@�檺�����Ȧs��mu(�榡1x9)
% @minus�����O-�����
% �]����������,mu�|�����N�x�}�X�i��������12x9����X_poly - mu
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

% @rdivide�h�O./�����
sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

% ���L�b�Ĥ@���@�~(ex1����featureNormalize.m)�ɧY�Ϥ���bsxfun�]�వ��@�˪����G
% ���T�w���ؤ�k����ٮį�N�O�F
% �]�γ\�O�ϥΪ��OOctave 5.2���~����12x9�M1x9���x�}�����۴���I���]�����w

% ============================================================

end
