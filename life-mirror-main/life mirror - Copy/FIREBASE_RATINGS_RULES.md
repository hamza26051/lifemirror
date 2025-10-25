# Updated Firebase Security Rules

## Complete Firebase Firestore security rules with userRatings collection:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {

    match /users/{userId} {
      allow read: if request.auth != null;
      allow write: if request.auth != null && (
        request.auth.uid == userId || 
        // Allow users to update follower/following counts when accepting follow requests
        (request.auth != null && (
          request.auth.uid == userId || 
          // Allow cross-user updates for follow system
          (resource == null || resource.data.followersCount != null || resource.data.followingCount != null)
        ))
      );

      // Follow requests
      match /followRequests/{requestId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }

      // Notifications under user
      match /notifications/{notificationId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }

      // Following under user - allow other users to add current user to their following list
      match /following/{followingId} {
        allow read: if request.auth != null;
        allow write: if request.auth != null && (
          request.auth.uid == userId || // User can write to their own following
          request.auth.uid == followingId // User can add themselves to someone else's following
        );
      }

      // Followers under user - allow other users to add current user to their followers list
      match /followers/{followerId} {
        allow read: if request.auth != null;
        allow write: if request.auth != null && (
          request.auth.uid == userId || // User can write to their own followers
          request.auth.uid == followerId // User can add themselves to someone else's followers
        );
      }
    }

    // Posts
    match /posts/{postId} {
      allow read, write: if request.auth != null;
    }

    // Global follow requests
    match /followRequests/{requestId} {
      allow read, write: if request.auth != null;
    }

    // Global notifications (if used)
    match /notifications/{notificationId} {
      allow read, write: if request.auth != null;
    }

    // Shared Results - NEW COLLECTION
    match /sharedResults/{resultId} {
      allow read: if true; // Anyone can read shared results
      allow write: if request.auth != null; // Only authenticated users can create
    }

    // User Ratings Collection - NEW COLLECTION
    match /userRatings/{ratingId} {
      allow read: if request.auth != null && request.auth.uid == resource.data.userId;
      allow write: if request.auth != null && request.auth.uid == request.resource.data.userId;
    }
  }
}
```

## What the userRatings rules do:

1. **Read Access**: Users can only read their own ratings
2. **Write Access**: Users can only create/update ratings for themselves
3. **Security**: Prevents users from accessing other users' ratings

## How to apply:

1. Go to Firebase Console
2. Navigate to Firestore Database
3. Click on "Rules" tab
4. Replace your existing rules with the complete rules above
5. Click "Publish"

## Collection Structure:

The `userRatings` collection stores:
- `userId`: The user who owns the rating
- `timestamp`: When the analysis was performed
- `overallAverage`: Overall average score
- `individualRatings`: All 6 individual attribute scores
- `analysisData`: Vibe analysis, improvement steps, etc.

This ensures all ratings are properly stored and can be retrieved with timestamps for the ratings history screen. 