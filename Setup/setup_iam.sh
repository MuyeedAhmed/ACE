#!/bin/bash
set -e

ROLE_NAME="ACE-EC2-S3-Role"
POLICY_NAME="ACE-S3-Bucket-Policy"
BUCKET_NAME="ma-njit-ace"

cat <<EOF > trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

cat <<EOF > s3-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        }
    ]
}
EOF

aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://trust-policy.json

POLICY_ARN=$(aws iam create-policy --policy-name "$POLICY_NAME" --policy-document file://s3-policy.json --query 'Policy.Arn' --output text)
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN"
aws iam create-instance-profile --instance-profile-name "$ROLE_NAME"
aws iam add-role-to-instance-profile --instance-profile-name "$ROLE_NAME" --role-name "$ROLE_NAME"
rm trust-policy.json s3-policy.json

echo "=========================================================="
echo " SUCCESS: IAM Role & Instance Profile setup completed!"
echo "=========================================================="
echo "Next step: Add '$ROLE_NAME' to 'IAM_INSTANCE_PROFILE' in your config.json."
